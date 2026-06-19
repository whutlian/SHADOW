from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shadow_hgc.ultra.papers100m_contract import make_t35_row
from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_stt_bank import load_selection_bank
from shadow_hgc.ultra.papers100m_teacher import load_teacher_topk_cache


def _ratio_dir_name(ratio: float) -> str:
    return f"ratio={float(ratio):.12g}".replace("+", "")


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", ".", "="} else "_" for ch in str(value))


def _condensed_dir(ctx: Papers100MCacheContext, ratio: float, *, policy: str | None = None, seed: int | None = None) -> Path:
    policy_value = _safe_component(str(ctx.selection_policy if policy is None else policy))
    seed_value = int(ctx.seed if seed is None else seed)
    return ctx.cache_root / "condensed" / f"policy={policy_value}_seed{seed_value}" / _ratio_dir_name(float(ratio))


def _legacy_condensed_dir(ctx: Papers100MCacheContext, ratio: float) -> Path:
    return ctx.cache_root / "condensed" / _ratio_dir_name(float(ratio))


def _concat_sft(cache_root: Path, target_size: int, feature_dim: int, rows: np.ndarray) -> np.ndarray:
    blocks = [
        np.memmap(cache_root / "sft" / "X0_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, feature_dim)),
        np.memmap(cache_root / "sft" / "X1_cite_ref_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, feature_dim)),
        np.memmap(cache_root / "sft" / "X1_cited_by_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, feature_dim)),
        np.memmap(cache_root / "sft" / "degree_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 2)),
        np.memmap(cache_root / "sft" / "label_support_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 1)),
        np.memmap(cache_root / "sft" / "label_entropy_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 1)),
    ]
    return np.concatenate([np.asarray(block[rows], dtype=np.float16) for block in blocks], axis=1)


def _dense_topk_targets(
    ids: np.ndarray,
    probs: np.ndarray,
    tail: np.ndarray,
    tail_prior: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    out = np.zeros((ids.shape[0], int(num_classes)), dtype=np.float32)
    out[np.arange(ids.shape[0])[:, None], ids] = probs
    prior = np.asarray(tail_prior, dtype=np.float32)
    prior = prior / max(float(prior.sum()), 1e-12)
    for row in range(ids.shape[0]):
        mask = np.ones(int(num_classes), dtype=bool)
        mask[ids[row]] = False
        denom = max(float(prior[mask].sum()), 1e-12)
        out[row, mask] = float(tail[row]) * prior[mask] / denom
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def _metrics(pred: np.ndarray, labels: np.ndarray, *, num_classes: int) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    mask = labels >= 0
    if not np.any(mask):
        return {"accuracy": 0.0, "macro_f1": 0.0, "predicted_classes": 0}
    pred = pred[mask]
    labels = labels[mask]
    accuracy = float((pred == labels).mean())
    f1_values = []
    for cls in range(int(num_classes)):
        tp = float(np.sum((pred == cls) & (labels == cls)))
        fp = float(np.sum((pred == cls) & (labels != cls)))
        fn = float(np.sum((pred != cls) & (labels == cls)))
        denom = 2.0 * tp + fp + fn
        f1_values.append(0.0 if denom <= 0 else (2.0 * tp / denom))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "predicted_classes": int(np.unique(pred).size),
    }


def train_and_eval_condensed_table(
    ctx: Papers100MCacheContext,
    ratio: float,
    *,
    student: str = "papers100m_gamlp_table",
    hidden_dim: int = 256,
    epochs: int = 220,
    temperature: float = 2.0,
    lambda_hard: float = 0.25,
    lambda_soft: float = 1.0,
    lambda_prior: float = 0.02,
    batch_size: int = 8192,
    eval_batch_size: int = 65536,
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = ctx.cache_root
    out_dir = _condensed_dir(ctx, float(ratio))
    if not (out_dir / "condensed_manifest.json").exists():
        legacy = _legacy_condensed_dir(ctx, float(ratio))
        if (legacy / "condensed_manifest.json").exists():
            out_dir = legacy
    manifest = read_json(out_dir / "condensed_manifest.json")
    num_rows = int(manifest["condensed_nodes"])
    z_shape = tuple(int(v) for v in manifest["z_shape"])
    num_classes = int(ctx.manifest["num_classes"])
    target_size = int(ctx.manifest["target_universe_size"])
    feature_dim = int(ctx.manifest["feature_dim"])
    topk = int(ctx.teacher["topk"])
    z = np.asarray(np.memmap(out_dir / "z_syn.fp16.memmap", mode="r", dtype=np.float16, shape=z_shape), dtype=np.float32)
    topk_ids = np.asarray(np.memmap(out_dir / "y_soft_topk_ids.u16.memmap", mode="r", dtype=np.uint16, shape=(num_rows, topk)), dtype=np.int64)
    topk_probs = np.asarray(np.memmap(out_dir / "y_soft_topk_probs.fp16.memmap", mode="r", dtype=np.float16, shape=(num_rows, topk)), dtype=np.float32)
    tail = np.asarray(np.memmap(out_dir / "y_soft_tail_mass.fp16.memmap", mode="r", dtype=np.float16, shape=(num_rows,)), dtype=np.float32)
    tail_prior = np.load(cache_root / "teacher" / ctx.teacher["tail_prior_path"]).astype(np.float32)
    y_soft = _dense_topk_targets(topk_ids, topk_probs, tail, tail_prior, num_classes=num_classes)
    hard = np.asarray(np.memmap(out_dir / "hard_anchor_labels.int16.memmap", mode="r", dtype=np.int16, shape=(num_rows,)), dtype=np.int64)
    target_device = torch.device("cuda" if str(device) == "auto" and torch.cuda.is_available() else ("cpu" if str(device) == "auto" else str(device)))
    torch.manual_seed(int(ctx.manifest.get("seed", 42) or 42))
    if str(student) == "papers100m_sagn_table":
        model = torch.nn.Sequential(
            torch.nn.Linear(z_shape[1], int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.10),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), num_classes),
        ).to(target_device)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(z_shape[1], int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), num_classes),
        ).to(target_device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    x = torch.from_numpy(z).to(target_device)
    y = torch.from_numpy(y_soft).to(target_device)
    hard_t = torch.from_numpy(hard).to(target_device)
    prior = torch.from_numpy(tail_prior / max(float(tail_prior.sum()), 1e-12)).to(target_device)
    temp = max(float(temperature), 1e-6)
    train_started = time.perf_counter()
    for epoch in range(int(epochs)):
        perm = torch.randperm(num_rows, device=target_device)
        for start in range(0, num_rows, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x[idx])
            logp = F.log_softmax(logits / temp, dim=1)
            loss = float(lambda_soft) * (-(y[idx] * logp).sum(dim=1).mean() * (temp * temp))
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

    def evaluate(local_path: str, size_key: str) -> dict[str, Any]:
        rows = np.asarray(np.memmap(cache_root / "raw" / local_path, mode="r", dtype=np.uint32, shape=(int(ctx.manifest[size_key]),)), dtype=np.int64)
        node_ids = np.asarray(np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))[rows], dtype=np.int64)
        labels = np.asarray(np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(ctx.manifest["num_nodes"]),))[node_ids], dtype=np.int64)
        preds = []
        model.eval()
        with torch.no_grad():
            for start in range(0, rows.size, int(eval_batch_size)):
                batch = rows[start : start + int(eval_batch_size)]
                feat = _concat_sft(cache_root, target_size, feature_dim, batch)
                logits = model(torch.from_numpy(np.asarray(feat, dtype=np.float32)).to(target_device))
                preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        return _metrics(np.concatenate(preds), labels, num_classes=num_classes)

    eval_started = time.perf_counter()
    valid = evaluate("valid_local_idx.u32.memmap", "valid_size")
    test = evaluate("test_local_idx.u32.memmap", "test_size")
    eval_time = time.perf_counter() - eval_started
    if target_device.type == "cuda":
        peak_gpu = int(torch.cuda.max_memory_allocated(target_device))
    else:
        peak_gpu = 0
    return {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "valid_acc": valid["accuracy"],
        "valid_macro_f1": valid["macro_f1"],
        "predicted_classes": test["predicted_classes"],
        "student_train_time": float(train_time),
        "eval_time": float(eval_time),
        "precompute_time": float(time.perf_counter() - started),
        "peak_gpu_ram": peak_gpu,
        "student": str(student),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "temperature": float(temperature),
        "lambda_hard": float(lambda_hard),
        "lambda_soft": float(lambda_soft),
        "lambda_prior": float(lambda_prior),
    }


def materialize_condensed_table(ctx: Papers100MCacheContext, ratio: float, *, policy: str = "stt_ratio_v2", seed: int = 42) -> dict[str, Any]:
    started = time.perf_counter()
    ctx.assert_ready(["manifest", "edge_cache", "sft_cache", "teacher_cache", "selection_bank"])
    cache_root = ctx.cache_root
    target_size = int(ctx.manifest["target_universe_size"])
    feature_dim = int(ctx.manifest["feature_dim"])
    denominator = int(ctx.manifest["num_nodes"])
    bank = load_selection_bank(cache_root, policy=policy, seed=seed)
    selected = bank.select_prefix(float(ratio), full_node_denominator=denominator)
    rows = np.asarray(selected, dtype=np.int64)
    out_dir = _condensed_dir(ctx, float(ratio), policy=policy, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = _concat_sft(cache_root, target_size, feature_dim, rows)
    z_mm = np.memmap(out_dir / "z_syn.fp16.memmap", mode="w+", dtype=np.float16, shape=z.shape)
    z_mm[:] = z
    z_mm.flush()
    del z_mm
    teacher = load_teacher_topk_cache(cache_root)
    topk_shape = (rows.size, int(teacher.manifest["topk"]))
    ids = np.memmap(out_dir / "y_soft_topk_ids.u16.memmap", mode="w+", dtype=np.uint16, shape=topk_shape)
    probs = np.memmap(out_dir / "y_soft_topk_probs.fp16.memmap", mode="w+", dtype=np.float16, shape=topk_shape)
    tail = np.memmap(out_dir / "y_soft_tail_mass.fp16.memmap", mode="w+", dtype=np.float16, shape=(rows.size,))
    ids[:] = teacher.topk_class_ids[rows]
    probs[:] = teacher.topk_probs[rows]
    tail[:] = teacher.tail_mass[rows]
    for mm in (ids, probs, tail):
        mm.flush()
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    source_ids = np.memmap(out_dir / "source_node_ids.u32.memmap", mode="w+", dtype=np.uint32, shape=(rows.size,))
    source_ids[:] = target_idx[rows]
    source_ids.flush()
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(denominator,))
    train_local = set(
        np.asarray(
            np.memmap(cache_root / "raw" / "train_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(ctx.manifest["train_size"]),)),
            dtype=np.int64,
        ).tolist()
    )
    hard_values = np.full(rows.size, -1, dtype=np.int16)
    train_mask = np.array([int(row) in train_local for row in rows], dtype=bool)
    if np.any(train_mask):
        hard_values[train_mask] = labels[np.asarray(source_ids, dtype=np.int64)[train_mask]]
    hard = np.memmap(out_dir / "hard_anchor_labels.int16.memmap", mode="w+", dtype=np.int16, shape=(rows.size,))
    hard[:] = hard_values
    hard.flush()
    weights = np.memmap(out_dir / "weights.fp16.memmap", mode="w+", dtype=np.float16, shape=(rows.size,))
    weights[:] = np.ones(rows.size, dtype=np.float16)
    weights.flush()
    condensed_bytes = directory_bytes(out_dir)
    ids_dict = ctx.cache_ids()
    manifest = {
        "ratio": float(ratio),
        "condensed_nodes": int(rows.size),
        "condensed_edges": 0,
        "z_shape": list(z.shape),
        "source_node_ids_shape": [int(rows.size)],
        "selection_policy": str(policy),
        "seed": int(seed),
        "condensed_dir": str(out_dir.relative_to(cache_root)),
        "teacher_cache_mode": ctx.teacher.get("teacher_cache_mode", ""),
        "parent_cache_ids": ids_dict,
        "condensed_cache_bytes": condensed_bytes,
        "condensed_materialize_time": float(time.perf_counter() - started),
        "created_at": utc_now(),
    }
    manifest["condensed_cache_id"] = stable_hash({"ratio": float(ratio), "policy": str(policy), "seed": int(seed), "parents": ids_dict, "rows": int(rows.size)})
    write_json(out_dir / "condensed_manifest.json", manifest)
    row = make_t35_row(
        requested_full_node_ratio=float(ratio),
        condensed_nodes=int(rows.size),
        full_node_ratio_denominator=denominator,
        target_universe_size=target_size,
        condensed_cache_bytes=condensed_bytes,
        condensed_materialize_time=manifest["condensed_materialize_time"],
        cache_root=str(cache_root),
        cache_build_id=ids_dict["cache_build_id"],
        edge_slice_cache_id=ids_dict["edge_slice_cache_id"],
        sft_cache_id=ids_dict["sft_cache_id"],
        teacher_cache_id=ids_dict["teacher_cache_id"],
        selection_bank_id=ids_dict["selection_bank_id"],
        teacher_cache_mode=ctx.teacher.get("teacher_cache_mode", ""),
        teacher_cache_bytes=ctx.teacher.get("teacher_cache_bytes", ""),
        teacher_dense_cache_bytes_diagnostic=ctx.teacher.get("teacher_dense_cache_bytes_diagnostic", ""),
        uses_dense_teacher_cache_in_ram=ctx.teacher.get("uses_dense_teacher_cache_in_ram", False),
        uses_teacher_probs_as_input=ctx.teacher.get("uses_teacher_probs_as_input", False),
        uses_teacher_probs_as_soft_targets=ctx.teacher.get("uses_teacher_probs_as_soft_targets", True),
        sft_cache_bytes=ctx.sft.get("sft_cache_bytes", ""),
        edge_cache_bytes=ctx.graph.get("edge_cache_bytes", ""),
        selection_bank_bytes=ctx.bank.get("selection_bank_bytes", ""),
        total_cache_bytes=directory_bytes(cache_root),
        num_nodes=denominator,
        num_edges=int(ctx.manifest["num_edges"]),
        num_classes=int(ctx.manifest["num_classes"]),
        feature_dim=feature_dim,
        train_size=int(ctx.manifest["train_size"]),
        valid_size=int(ctx.manifest["valid_size"]),
        test_size=int(ctx.manifest["test_size"]),
        sft_block_manifest="sft/sft_manifest.json",
        selection_policy=policy,
        nested_selection=bool(ctx.bank.get("nested_selection", True)),
        bucket_core_count=ctx.bank.get("bucket_core_count", 0),
        bucket_boundary_count=ctx.bank.get("bucket_boundary_count", 0),
        bucket_rare_count=ctx.bank.get("bucket_rare_count", 0),
        bucket_prior_repair_count=ctx.bank.get("bucket_prior_repair_count", 0),
        bucket_hard_anchor_count=ctx.bank.get("bucket_hard_anchor_count", 0),
    )
    return row
