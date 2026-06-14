from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shadow_hgc.ultra.papers100m_condensed import _concat_sft, _dense_topk_targets, _metrics
from shadow_hgc.ultra.papers100m_memmap import read_json
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext


def weighted_sgc_propagate(
    x: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray | None = None,
    *,
    num_hops: int = 1,
    normalize_dst: bool = True,
    add_self: bool = True,
) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)
    if edge_weight is None:
        weight = np.ones(src.shape[0], dtype=np.float32)
    else:
        weight = np.asarray(edge_weight, dtype=np.float32)
    if np.any(weight < 0):
        raise ValueError("SGC backend requires nonnegative edge weights")
    for _ in range(int(num_hops)):
        w = weight
        if normalize_dst and w.size:
            denom = np.bincount(dst, weights=w, minlength=h.shape[0]).astype(np.float32)
            w = w / np.maximum(denom[dst], 1e-12)
        out = np.zeros_like(h, dtype=np.float32)
        if src.size:
            np.add.at(out, dst, h[src] * w[:, None])
        if add_self:
            out = out + h
        h = out
    return h


def _load_ant_edges(edge_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(edge_dir)
    manifest = read_json(root / "ant_manifest.json")
    edge_count = int(manifest["ant_edges"])
    src = np.asarray(np.memmap(root / "edge_src.u32.memmap", mode="r", dtype=np.uint32, shape=(edge_count,)), dtype=np.int64)
    dst = np.asarray(np.memmap(root / "edge_dst.u32.memmap", mode="r", dtype=np.uint32, shape=(edge_count,)), dtype=np.int64)
    weight = np.asarray(np.memmap(root / "edge_weight.fp16.memmap", mode="r", dtype=np.float16, shape=(edge_count,)), dtype=np.float32)
    return src, dst, weight


def train_and_eval_sgc_condensed(
    ctx: Papers100MCacheContext,
    ratio: float,
    *,
    hidden_dim: int = 0,
    epochs: int = 180,
    temperature: float = 1.5,
    lambda_hard: float = 0.75,
    lambda_prior: float = 0.02,
    batch_size: int = 8192,
    eval_batch_size: int = 65536,
    device: str = "auto",
    ant_edge_dir: str | Path | None = None,
    sgc_hops: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = ctx.cache_root
    ratio_dir = f"ratio={float(ratio):.12g}".replace("+", "")
    out_dir = cache_root / "condensed" / ratio_dir
    manifest = read_json(out_dir / "condensed_manifest.json")
    num_rows = int(manifest["condensed_nodes"])
    z_shape = tuple(int(v) for v in manifest["z_shape"])
    target_size = int(ctx.manifest["target_universe_size"])
    feature_dim = int(ctx.manifest["feature_dim"])
    num_classes = int(ctx.manifest["num_classes"])
    topk = int(ctx.teacher["topk"])
    z = np.asarray(np.memmap(out_dir / "z_syn.fp16.memmap", mode="r", dtype=np.float16, shape=z_shape), dtype=np.float32)
    ant_edges = 0
    if ant_edge_dir is not None:
        src, dst, weight = _load_ant_edges(ant_edge_dir)
        z = weighted_sgc_propagate(z, src, dst, weight, num_hops=sgc_hops, normalize_dst=True, add_self=True)
        ant_edges = int(src.size)
    topk_ids = np.asarray(np.memmap(out_dir / "y_soft_topk_ids.u16.memmap", mode="r", dtype=np.uint16, shape=(num_rows, topk)), dtype=np.int64)
    topk_probs = np.asarray(np.memmap(out_dir / "y_soft_topk_probs.fp16.memmap", mode="r", dtype=np.float16, shape=(num_rows, topk)), dtype=np.float32)
    tail = np.asarray(np.memmap(out_dir / "y_soft_tail_mass.fp16.memmap", mode="r", dtype=np.float16, shape=(num_rows,)), dtype=np.float32)
    tail_prior = np.load(cache_root / "teacher" / ctx.teacher["tail_prior_path"]).astype(np.float32)
    y_soft = _dense_topk_targets(topk_ids, topk_probs, tail, tail_prior, num_classes=num_classes)
    hard = np.asarray(np.memmap(out_dir / "hard_anchor_labels.int16.memmap", mode="r", dtype=np.int16, shape=(num_rows,)), dtype=np.int64)
    target_device = torch.device("cuda" if str(device) == "auto" and torch.cuda.is_available() else ("cpu" if str(device) == "auto" else str(device)))
    torch.manual_seed(int(ctx.seed))
    if int(hidden_dim) > 0:
        model = torch.nn.Sequential(
            torch.nn.Linear(z_shape[1], int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), num_classes),
        ).to(target_device)
    else:
        model = torch.nn.Linear(z_shape[1], num_classes).to(target_device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    x = torch.from_numpy(z).to(target_device)
    y = torch.from_numpy(y_soft).to(target_device)
    hard_t = torch.from_numpy(hard).to(target_device)
    prior = torch.from_numpy(tail_prior / max(float(tail_prior.sum()), 1e-12)).to(target_device)
    temp = max(float(temperature), 1e-6)
    train_started = time.perf_counter()
    model.train()
    for _epoch in range(int(epochs)):
        perm = torch.randperm(num_rows, device=target_device)
        for start in range(0, num_rows, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x[idx])
            loss = -(y[idx] * F.log_softmax(logits / temp, dim=1)).sum(dim=1).mean() * (temp * temp)
            mask = hard_t[idx] >= 0
            if bool(mask.any().item()) and float(lambda_hard) != 0.0:
                loss = loss + float(lambda_hard) * F.cross_entropy(logits[mask], hard_t[idx][mask].long())
            if float(lambda_prior) != 0.0:
                pred_prior = torch.softmax(logits.float(), dim=1).mean(dim=0).clamp_min(1e-12)
                loss = loss + float(lambda_prior) * F.kl_div(pred_prior.log(), prior, reduction="sum")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - train_started

    def evaluate(split: str) -> dict[str, Any]:
        rows = np.asarray(
            np.memmap(cache_root / "raw" / f"{split}_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(ctx.manifest[f"{split}_size"]),)),
            dtype=np.int64,
        )
        target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
        labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(ctx.manifest["num_nodes"]),))
        node_ids = np.asarray(target_idx[rows], dtype=np.int64)
        split_labels = np.asarray(labels[node_ids], dtype=np.int64)
        preds = []
        model.eval()
        with torch.no_grad():
            for start in range(0, rows.size, int(eval_batch_size)):
                batch = rows[start : start + int(eval_batch_size)]
                feat = _concat_sft(cache_root, target_size, feature_dim, batch)
                logits = model(torch.from_numpy(np.asarray(feat, dtype=np.float32)).to(target_device))
                preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        return _metrics(np.concatenate(preds), split_labels, num_classes=num_classes)

    eval_started = time.perf_counter()
    valid = evaluate("valid")
    test = evaluate("test")
    eval_time = time.perf_counter() - eval_started
    peak_gpu = int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0
    return {
        "backend": "sgc",
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "valid_acc": valid["accuracy"],
        "valid_macro_f1": valid["macro_f1"],
        "predicted_classes": test["predicted_classes"],
        "student_train_time": float(train_time),
        "eval_time": float(eval_time),
        "precompute_time": float(time.perf_counter() - started),
        "peak_gpu_ram": peak_gpu,
        "student": "papers100m_sgc_student_on_condensed",
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "temperature": float(temperature),
        "lambda_prior": float(lambda_prior),
        "ant_edges": ant_edges,
    }
