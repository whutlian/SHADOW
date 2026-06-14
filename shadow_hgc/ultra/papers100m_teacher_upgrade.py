from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shadow_hgc.ultra.papers100m_condensed import _concat_sft
from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json
from shadow_hgc.ultra.papers100m_teacher import _classification_metrics, _k_from_mode, _topk_payload_from_logits


TEACHER_ALIASES = {
    "sgc": "papers100m_sgc_teacher_v2",
    "sign": "papers100m_sign_teacher_v2",
    "sagn_lite": "papers100m_sagn_lite_teacher_v2",
    "gamlp_lite": "papers100m_gamlp_lite_teacher_v2",
    "ensemble": "papers100m_teacher_ensemble_v2",
}


@dataclass(frozen=True)
class TeacherFeatureSpec:
    mode: str
    input_dim: int
    missing_blocks: tuple[str, ...]
    notes: str


def resolve_feature_spec(cache_root: str | Path, mode: str) -> TeacherFeatureSpec:
    cache_root = Path(cache_root)
    manifest = read_json(cache_root / "manifest.json")
    feature_dim = int(manifest["feature_dim"])
    base_dim = feature_dim * 3 + 4
    mode = str(mode)
    if mode == "minimal":
        return TeacherFeatureSpec(mode=mode, input_dim=base_dim, missing_blocks=(), notes="")
    if mode == "v2":
        missing = []
        extra_dim = 0
        for name in ("X2_cite_ref_target.fp16.memmap", "X2_cited_by_target.fp16.memmap", "Xres_target.fp16.memmap", "label_max_affinity_target.fp16.memmap"):
            if (cache_root / "sft" / name).exists():
                extra_dim += feature_dim if name.startswith("X") else 1
            else:
                missing.append(name)
        return TeacherFeatureSpec(
            mode=mode,
            input_dim=base_dim + extra_dim,
            missing_blocks=tuple(missing),
            notes="v2_missing_blocks_fallback_to_available" if missing else "",
        )
    if mode == "temporal":
        missing = []
        extra_dim = 0
        for name in ("year_target.fp16.memmap", "year_bucket_target.fp16.memmap", "temporal_train_prior_target.fp16.memmap"):
            if (cache_root / "sft" / name).exists():
                extra_dim += 1
            else:
                missing.append(name)
        return TeacherFeatureSpec(
            mode=mode,
            input_dim=base_dim + extra_dim,
            missing_blocks=tuple(missing),
            notes="temporal_missing_blocks_fallback_to_minimal" if missing else "",
        )
    raise ValueError(f"unknown feature block mode: {mode}")


def concat_teacher_features(cache_root: str | Path, rows: np.ndarray, spec: TeacherFeatureSpec) -> np.ndarray:
    cache_root = Path(cache_root)
    manifest = read_json(cache_root / "manifest.json")
    target_size = int(manifest["target_universe_size"])
    feature_dim = int(manifest["feature_dim"])
    parts = [_concat_sft(cache_root, target_size, feature_dim, np.asarray(rows, dtype=np.int64))]
    if spec.mode == "v2":
        for name in ("X2_cite_ref_target.fp16.memmap", "X2_cited_by_target.fp16.memmap", "Xres_target.fp16.memmap"):
            path = cache_root / "sft" / name
            if path.exists():
                block = np.memmap(path, mode="r", dtype=np.float16, shape=(target_size, feature_dim))
                parts.append(np.asarray(block[rows], dtype=np.float16))
        path = cache_root / "sft" / "label_max_affinity_target.fp16.memmap"
        if path.exists():
            block = np.memmap(path, mode="r", dtype=np.float16, shape=(target_size, 1))
            parts.append(np.asarray(block[rows], dtype=np.float16))
    elif spec.mode == "temporal":
        for name in ("year_target.fp16.memmap", "year_bucket_target.fp16.memmap", "temporal_train_prior_target.fp16.memmap"):
            path = cache_root / "sft" / name
            if path.exists():
                block = np.memmap(path, mode="r", dtype=np.float16, shape=(target_size, 1))
                parts.append(np.asarray(block[rows], dtype=np.float16))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _teacher_model(method: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    method_id = TEACHER_ALIASES.get(str(method), str(method))
    if method_id == "papers100m_sgc_teacher_v2":
        return torch.nn.Linear(input_dim, num_classes)
    if method_id == "papers100m_sign_teacher_v2":
        return torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(384, num_classes),
        )
    if method_id == "papers100m_sagn_lite_teacher_v2":
        return torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes),
        )
    if method_id == "papers100m_gamlp_lite_teacher_v2":
        return torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(512, num_classes),
        )
    raise ValueError(f"unsupported teacher method: {method}")


def _split_rows_and_labels(cache_root: Path, split: str, manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target_size = int(manifest["target_universe_size"])
    rows = np.asarray(
        np.memmap(cache_root / "raw" / f"{split}_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(manifest[f"{split}_size"]),)),
        dtype=np.int64,
    )
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(manifest["num_nodes"]),))
    node_ids = np.asarray(target_idx[rows], dtype=np.int64)
    return rows, np.asarray(labels[node_ids], dtype=np.int64)


def _evaluate_model(
    cache_root: Path,
    model: torch.nn.Module,
    spec: TeacherFeatureSpec,
    *,
    split: str,
    manifest: dict[str, Any],
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    rows, labels = _split_rows_and_labels(cache_root, split, manifest)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            batch = rows[start : start + int(batch_size)]
            feat = concat_teacher_features(cache_root, batch, spec)
            logits = model(torch.from_numpy(feat).to(device))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return _classification_metrics(np.concatenate(preds), labels, num_classes=num_classes)


def _write_model_topk_cache(
    cache_root: Path,
    teacher_dir: Path,
    model: torch.nn.Module,
    spec: TeacherFeatureSpec,
    *,
    mode: str,
    manifest: dict[str, Any],
    batch_size: int,
    device: torch.device,
    teacher_train_time: float,
    valid: dict[str, Any],
    test: dict[str, Any],
    teacher_id: str,
    method_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    teacher_dir.mkdir(parents=True, exist_ok=True)
    target_size = int(manifest["target_universe_size"])
    num_classes = int(manifest["num_classes"])
    k = min(_k_from_mode(mode), num_classes)
    ids_mm = np.memmap(teacher_dir / f"topk{k}_class_ids.u16.memmap", mode="w+", dtype=np.uint16, shape=(target_size, k))
    probs_mm = np.memmap(teacher_dir / f"topk{k}_probs.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size, k))
    tail_mm = np.memmap(teacher_dir / "tail_mass.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size,))
    entropy_mm = np.memmap(teacher_dir / "entropy.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size,))
    margin_mm = np.memmap(teacher_dir / "margin.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size,))
    prior_sum = np.zeros(num_classes, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, target_size, int(batch_size)):
            stop = min(start + int(batch_size), target_size)
            rows = np.arange(start, stop, dtype=np.int64)
            feat = concat_teacher_features(cache_root, rows, spec)
            logits = model(torch.from_numpy(feat).to(device)).detach().cpu().numpy()
            ids, vals, tail, entropy, margin, chunk_prior = _topk_payload_from_logits(logits, k=k)
            ids_mm[start:stop] = ids
            probs_mm[start:stop] = vals
            tail_mm[start:stop] = tail
            entropy_mm[start:stop] = entropy
            margin_mm[start:stop] = margin
            prior_sum += chunk_prior
    for mm in (ids_mm, probs_mm, tail_mm, entropy_mm, margin_mm):
        mm.flush()
    del ids_mm, probs_mm, tail_mm, entropy_mm, margin_mm
    prior = (prior_sum / max(1, target_size)).astype(np.float32)
    np.save(teacher_dir / "tail_prior.fp32.npy", prior)
    sft_manifest = read_json(cache_root / "sft" / "sft_manifest.json")
    payload = {
        "method": method_id,
        "teacher_id": teacher_id,
        "dataset_name": manifest["dataset_name"],
        "cache_build_id": manifest.get("cache_build_id", ""),
        "edge_cache_id": read_json(cache_root / "graph" / "edge_slice_manifest.json").get("edge_slice_cache_id", ""),
        "sft_cache_id": sft_manifest.get("sft_cache_id", ""),
        "target_universe_size": target_size,
        "num_classes": num_classes,
        "feature_block_mode": spec.mode,
        "missing_feature_blocks": list(spec.missing_blocks),
        "teacher_cache_scope": "target_universe",
        "teacher_cache_mode": str(mode),
        "topk": k,
        "topk_class_ids_path": f"topk{k}_class_ids.u16.memmap",
        "topk_probs_path": f"topk{k}_probs.fp16.memmap",
        "tail_mass_path": "tail_mass.fp16.memmap",
        "entropy_path": "entropy.fp16.memmap",
        "margin_path": "margin.fp16.memmap",
        "tail_prior_path": "tail_prior.fp32.npy",
        "teacher_topk_build_mode": "streaming_logits",
        "uses_streaming_logits": True,
        "uses_dense_teacher_cache_in_ram": False,
        "uses_dense_all_node_teacher_cache": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_teacher_probs_as_input": False,
        "uses_teacher_probs_as_soft_targets": True,
        "teacher_train_time": float(teacher_train_time),
        "teacher_infer_time": float(time.perf_counter() - started),
        "valid_acc": valid["accuracy"],
        "valid_macro_f1": valid["macro_f1"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "predicted_classes": test["predicted_classes"],
        "teacher_cache_bytes": directory_bytes(teacher_dir),
        "created_at": utc_now(),
    }
    payload["teacher_cache_id"] = stable_hash(
        {
            "method": method_id,
            "mode": mode,
            "feature_mode": spec.mode,
            "rows": target_size,
            "classes": num_classes,
            "valid": round(float(valid["accuracy"]), 6),
            "test": round(float(test["accuracy"]), 6),
        }
    )
    write_json(teacher_dir / "teacher_cache_manifest.json", payload)
    write_json(teacher_dir / "teacher_metrics.json", payload)
    return payload


def train_teacher_upgrade(
    cache_root: str | Path,
    *,
    method: str,
    feature_block_mode: str = "minimal",
    teacher_cache_mode: str = "topk8_tail",
    seed: int = 7,
    epochs: int = 80,
    batch_size: int = 8192,
    eval_batch_size: int = 65536,
    infer_batch_size: int = 65536,
    device: str = "auto",
    force: bool = False,
    preload_train: bool = False,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    manifest = read_json(cache_root / "manifest.json")
    method_id = TEACHER_ALIASES.get(str(method), str(method))
    spec = resolve_feature_spec(cache_root, feature_block_mode)
    teacher_id = f"{method_id}_{spec.mode}_{teacher_cache_mode}_seed{int(seed)}"
    teacher_dir = cache_root / "teacher_upgrade" / teacher_id
    manifest_path = teacher_dir / "teacher_cache_manifest.json"
    if manifest_path.exists() and not force:
        payload = read_json(manifest_path)
        return teacher_row_from_manifest(payload, peak_gpu_ram=0, promotion_status=None)
    if method_id == "papers100m_teacher_ensemble_v2":
        return {
            "method": method_id,
            "teacher_id": teacher_id,
            "feature_block_mode": spec.mode,
            "teacher_cache_mode": teacher_cache_mode,
            "promotion_status": "diagnostic",
            "failure_reason": "ensemble_requires_existing_candidate_caches",
            "uses_streaming_logits": True,
            "uses_dense_teacher_cache_in_ram": False,
            "uses_dense_all_node_teacher_cache": False,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
        }
    target_device = torch.device("cuda" if str(device) == "auto" and torch.cuda.is_available() else ("cpu" if str(device) == "auto" else str(device)))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    num_classes = int(manifest["num_classes"])
    train_rows, train_labels = _split_rows_and_labels(cache_root, "train", manifest)
    model = _teacher_model(method_id, spec.input_dim, num_classes).to(target_device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_x_t: torch.Tensor | None = None
    train_y_t: torch.Tensor | None = None
    if bool(preload_train):
        valid_mask = train_labels >= 0
        train_rows = train_rows[valid_mask]
        train_labels = train_labels[valid_mask]
        train_x_np = concat_teacher_features(cache_root, train_rows, spec)
        train_x_t = torch.from_numpy(train_x_np).to(target_device)
        train_y_t = torch.from_numpy(train_labels.astype(np.int64)).to(target_device)
    train_started = time.perf_counter()
    for _epoch in range(int(epochs)):
        if train_x_t is not None:
            perm_t = torch.randperm(int(train_x_t.shape[0]), device=target_device)
            model.train()
            for start in range(0, int(train_x_t.shape[0]), int(batch_size)):
                idx = perm_t[start : start + int(batch_size)]
                logits = model(train_x_t[idx])
                loss = F.cross_entropy(logits, train_y_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            continue
        perm = np.random.permutation(train_rows.size)
        model.train()
        for start in range(0, train_rows.size, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            rows = train_rows[idx]
            y = train_labels[idx]
            mask = y >= 0
            if not np.any(mask):
                continue
            feat = concat_teacher_features(cache_root, rows[mask], spec)
            logits = model(torch.from_numpy(feat).to(target_device))
            target = torch.from_numpy(y[mask].astype(np.int64)).to(target_device)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - train_started
    valid = _evaluate_model(cache_root, model, spec, split="valid", manifest=manifest, num_classes=num_classes, batch_size=eval_batch_size, device=target_device)
    test = _evaluate_model(cache_root, model, spec, split="test", manifest=manifest, num_classes=num_classes, batch_size=eval_batch_size, device=target_device)
    payload = _write_model_topk_cache(
        cache_root,
        teacher_dir,
        model,
        spec,
        mode=teacher_cache_mode,
        manifest=manifest,
        batch_size=infer_batch_size,
        device=target_device,
        teacher_train_time=train_time,
        valid=valid,
        test=test,
        teacher_id=teacher_id,
        method_id=method_id,
    )
    peak_gpu = int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0
    return teacher_row_from_manifest(payload, peak_gpu_ram=peak_gpu, promotion_status=None)


def teacher_row_from_manifest(payload: dict[str, Any], *, peak_gpu_ram: int = 0, promotion_status: str | None = None) -> dict[str, Any]:
    test_acc = float(payload.get("accuracy", payload.get("test_acc", 0.0)) or 0.0)
    status = promotion_status
    failure = ""
    if status is None:
        if test_acc >= 0.60:
            status = "promoted"
        elif test_acc >= 0.55:
            status = "promoted_first_gate"
        else:
            status = "diagnostic"
            failure = "teacher_below_0p55_gate"
    return {
        "method": payload.get("method", ""),
        "teacher_id": payload.get("teacher_id", ""),
        "cache_build_id": payload.get("cache_build_id", ""),
        "edge_cache_id": payload.get("edge_cache_id", ""),
        "sft_cache_id": payload.get("sft_cache_id", ""),
        "feature_block_mode": payload.get("feature_block_mode", ""),
        "uses_streaming_logits": payload.get("uses_streaming_logits", True),
        "teacher_cache_mode": payload.get("teacher_cache_mode", ""),
        "teacher_cache_id": payload.get("teacher_cache_id", ""),
        "valid_acc": payload.get("valid_acc", ""),
        "test_acc": payload.get("accuracy", ""),
        "macro_f1": payload.get("macro_f1", ""),
        "predicted_classes": payload.get("predicted_classes", ""),
        "topk_cache_bytes": payload.get("teacher_cache_bytes", ""),
        "train_time": payload.get("teacher_train_time", ""),
        "infer_time": payload.get("teacher_infer_time", ""),
        "peak_cpu_ram": "",
        "peak_gpu_ram": peak_gpu_ram,
        "uses_dense_teacher_cache_in_ram": payload.get("uses_dense_teacher_cache_in_ram", False),
        "uses_dense_all_node_teacher_cache": payload.get("uses_dense_all_node_teacher_cache", False),
        "uses_valid_labels_as_input": payload.get("uses_valid_labels_as_input", False),
        "uses_test_labels_as_input": payload.get("uses_test_labels_as_input", False),
        "promotion_status": status,
        "failure_reason": failure,
        "notes": payload.get("missing_feature_blocks", ""),
    }


def install_teacher_upgrade(cache_root: str | Path, teacher_id: str) -> dict[str, Any]:
    cache_root = Path(cache_root)
    src = cache_root / "teacher_upgrade" / str(teacher_id)
    if not (src / "teacher_cache_manifest.json").exists():
        raise FileNotFoundError(f"teacher upgrade cache does not exist: {src}")
    dst = cache_root / "teacher"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            shutil.copy2(path, dst / path.name)
    payload = read_json(dst / "teacher_cache_manifest.json")
    payload["installed_from_teacher_upgrade"] = str(teacher_id)
    write_json(dst / "teacher_cache_manifest.json", payload)
    return payload
