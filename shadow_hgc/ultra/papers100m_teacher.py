from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.sft.stt_cache import estimate_stt_cache_bytes
from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json


def _k_from_mode(mode: str) -> int:
    digits = "".join(ch for ch in str(mode) if ch.isdigit())
    return int(digits) if digits else 8


def _as_probs(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    row_sum = x.sum(axis=1, keepdims=True)
    if np.all(x >= 0) and np.allclose(row_sum, np.ones_like(row_sum), atol=1e-4):
        return x / np.maximum(row_sum, 1e-12)
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _topk_payload_from_logits(logits: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = _softmax_logits(logits)
    num_rows, num_classes = int(probs.shape[0]), int(probs.shape[1])
    k = min(int(k), num_classes)
    if k == num_classes:
        order = np.argsort(-probs, axis=1)
    else:
        candidates = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
        candidate_probs = np.take_along_axis(probs, candidates, axis=1)
        order = np.take_along_axis(candidates, np.argsort(-candidate_probs, axis=1), axis=1)
    vals = np.take_along_axis(probs, order, axis=1)
    tail = np.maximum(1.0 - vals.sum(axis=1), 0.0)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    margin = vals[:, 0] - vals[:, 1] if k >= 2 else vals[:, 0]
    prior_sum = probs.sum(axis=0)
    assert order.shape == (num_rows, k)
    return order.astype(np.uint16), vals.astype(np.float16), tail.astype(np.float16), entropy.astype(np.float16), margin.astype(np.float16), prior_sum.astype(np.float64)


def _classification_metrics(pred: np.ndarray, labels: np.ndarray, *, num_classes: int) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    mask = labels >= 0
    if not np.any(mask):
        return {"accuracy": 0.0, "macro_f1": 0.0, "predicted_classes": 0}
    pred = pred[mask]
    labels = labels[mask]
    f1_values = []
    for cls in range(int(num_classes)):
        tp = float(np.sum((pred == cls) & (labels == cls)))
        fp = float(np.sum((pred == cls) & (labels != cls)))
        fn = float(np.sum((pred != cls) & (labels == cls)))
        denom = 2.0 * tp + fp + fn
        f1_values.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "predicted_classes": int(np.unique(pred).size),
    }


@dataclass
class Papers100MTopKTeacherCache:
    root: Path
    manifest: dict[str, Any]
    topk_class_ids: np.memmap
    topk_probs: np.memmap
    tail_mass: np.memmap
    tail_prior: np.ndarray
    entropy: np.memmap
    margin: np.memmap

    def reconstruct_rows(self, rows: np.ndarray, *, num_classes: int) -> np.ndarray:
        idx = np.asarray(rows, dtype=np.int64)
        ids = np.asarray(self.topk_class_ids[idx], dtype=np.int64)
        vals = np.asarray(self.topk_probs[idx], dtype=np.float32)
        out = np.zeros((idx.size, int(num_classes)), dtype=np.float32)
        batch = np.arange(idx.size)[:, None]
        out[batch, ids] = vals
        prior = np.asarray(self.tail_prior, dtype=np.float32)
        prior = prior / max(float(prior.sum()), 1e-12)
        tail = np.asarray(self.tail_mass[idx], dtype=np.float32)
        for row in range(idx.size):
            mask = np.ones(int(num_classes), dtype=bool)
            mask[ids[row]] = False
            denom = max(float(prior[mask].sum()), 1e-12)
            out[row, mask] = tail[row] * prior[mask] / denom
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def write_teacher_topk_cache_from_probs(cache_root: str | Path, probs_or_logits: np.ndarray, *, mode: str = "topk8_tail") -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = Path(cache_root)
    teacher_dir = cache_root / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    probs = _as_probs(probs_or_logits)
    num_rows, num_classes = int(probs.shape[0]), int(probs.shape[1])
    k = min(_k_from_mode(mode), num_classes)
    order = np.argsort(-probs, axis=1)[:, :k]
    vals = np.take_along_axis(probs, order, axis=1)
    tail = np.maximum(1.0 - vals.sum(axis=1), 0.0)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    if k >= 2:
        margin = vals[:, 0] - vals[:, 1]
    else:
        margin = vals[:, 0]
    prior = probs.mean(axis=0).astype(np.float32)

    ids_mm = np.memmap(teacher_dir / f"topk{k}_class_ids.u16.memmap", mode="w+", dtype=np.uint16, shape=(num_rows, k))
    vals_mm = np.memmap(teacher_dir / f"topk{k}_probs.fp16.memmap", mode="w+", dtype=np.float16, shape=(num_rows, k))
    tail_mm = np.memmap(teacher_dir / "tail_mass.fp16.memmap", mode="w+", dtype=np.float16, shape=(num_rows,))
    entropy_mm = np.memmap(teacher_dir / "entropy.fp16.memmap", mode="w+", dtype=np.float16, shape=(num_rows,))
    margin_mm = np.memmap(teacher_dir / "margin.fp16.memmap", mode="w+", dtype=np.float16, shape=(num_rows,))
    ids_mm[:] = order.astype(np.uint16)
    vals_mm[:] = vals.astype(np.float16)
    tail_mm[:] = tail.astype(np.float16)
    entropy_mm[:] = entropy.astype(np.float16)
    margin_mm[:] = margin.astype(np.float16)
    for mm in (ids_mm, vals_mm, tail_mm, entropy_mm, margin_mm):
        mm.flush()
    del ids_mm, vals_mm, tail_mm, entropy_mm, margin_mm
    np.save(teacher_dir / "tail_prior.fp32.npy", prior)
    estimates = estimate_stt_cache_bytes(num_nodes=num_rows, num_classes=num_classes, mode=str(mode))
    manifest = {
        "dataset_name": read_json(cache_root / "manifest.json").get("dataset_name", "ogbn-papers100M") if (cache_root / "manifest.json").exists() else "ogbn-papers100M",
        "target_universe_size": num_rows,
        "num_classes": num_classes,
        "teacher_cache_scope": "target_universe",
        "teacher_cache_mode": str(mode),
        "topk": k,
        "topk_class_ids_path": f"topk{k}_class_ids.u16.memmap",
        "topk_probs_path": f"topk{k}_probs.fp16.memmap",
        "tail_mass_path": "tail_mass.fp16.memmap",
        "entropy_path": "entropy.fp16.memmap",
        "margin_path": "margin.fp16.memmap",
        "tail_prior_path": "tail_prior.fp32.npy",
        "teacher_cache_bytes": directory_bytes(teacher_dir),
        "teacher_dense_cache_bytes_diagnostic": estimates["teacher_dense_cache_bytes_diagnostic"],
        "uses_dense_all_node_teacher_cache": False,
        "uses_dense_teacher_cache_in_ram": True,
        "uses_teacher_probs_as_input": False,
        "uses_teacher_probs_as_soft_targets": True,
        "teacher_topk_build_mode": "supplied_dense_probs_or_logits",
        "teacher_train_time": 0.0,
        "teacher_infer_time": float(time.perf_counter() - started),
        "valid_acc": "",
        "accuracy": "",
        "macro_f1": "",
        "created_at": utc_now(),
    }
    manifest["teacher_cache_id"] = stable_hash({"mode": mode, "rows": num_rows, "classes": num_classes, "prior": prior.round(6).tolist()})
    write_json(teacher_dir / "teacher_cache_manifest.json", manifest)
    write_json(teacher_dir / "teacher_config.json", {"teacher": "prototype_or_supplied_probs", "mode": mode})
    write_json(teacher_dir / "teacher_metrics.json", {key: manifest[key] for key in ("valid_acc", "accuracy", "macro_f1", "teacher_train_time", "teacher_infer_time")})
    return manifest


def write_teacher_topk_cache_from_prototypes(
    cache_root: str | Path,
    features: np.memmap,
    prototypes: np.ndarray,
    *,
    class_bias: np.ndarray | None = None,
    mode: str = "topk8_tail",
    chunk_size: int = 65_536,
    teacher_train_time: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = Path(cache_root)
    teacher_dir = cache_root / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    target_size = int(features.shape[0])
    num_classes = int(prototypes.shape[0])
    k = min(_k_from_mode(mode), num_classes)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    bias = np.asarray(class_bias, dtype=np.float32) if class_bias is not None else None

    ids_mm = np.memmap(teacher_dir / f"topk{k}_class_ids.u16.memmap", mode="w+", dtype=np.uint16, shape=(target_size, k))
    vals_mm = np.memmap(teacher_dir / f"topk{k}_probs.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size, k))
    tail_mm = np.memmap(teacher_dir / "tail_mass.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size,))
    entropy_mm = np.memmap(teacher_dir / "entropy.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size,))
    margin_mm = np.memmap(teacher_dir / "margin.fp16.memmap", mode="w+", dtype=np.float16, shape=(target_size,))
    prior_sum = np.zeros(num_classes, dtype=np.float64)
    for start in range(0, target_size, int(chunk_size)):
        stop = min(start + int(chunk_size), target_size)
        logits = np.asarray(features[start:stop], dtype=np.float32) @ prototypes.T
        if bias is not None:
            logits += bias[None, :]
        ids, vals, tail, entropy, margin, chunk_prior = _topk_payload_from_logits(logits, k=k)
        ids_mm[start:stop] = ids
        vals_mm[start:stop] = vals
        tail_mm[start:stop] = tail
        entropy_mm[start:stop] = entropy
        margin_mm[start:stop] = margin
        prior_sum += chunk_prior
    for mm in (ids_mm, vals_mm, tail_mm, entropy_mm, margin_mm):
        mm.flush()
    del ids_mm, vals_mm, tail_mm, entropy_mm, margin_mm
    prior = (prior_sum / max(1, target_size)).astype(np.float32)
    np.save(teacher_dir / "tail_prior.fp32.npy", prior)
    estimates = estimate_stt_cache_bytes(num_nodes=target_size, num_classes=num_classes, mode=str(mode))
    manifest = {
        "dataset_name": read_json(cache_root / "manifest.json").get("dataset_name", "ogbn-papers100M") if (cache_root / "manifest.json").exists() else "ogbn-papers100M",
        "target_universe_size": target_size,
        "num_classes": num_classes,
        "teacher_cache_scope": "target_universe",
        "teacher_cache_mode": str(mode),
        "topk": k,
        "topk_class_ids_path": f"topk{k}_class_ids.u16.memmap",
        "topk_probs_path": f"topk{k}_probs.fp16.memmap",
        "tail_mass_path": "tail_mass.fp16.memmap",
        "entropy_path": "entropy.fp16.memmap",
        "margin_path": "margin.fp16.memmap",
        "tail_prior_path": "tail_prior.fp32.npy",
        "teacher_cache_bytes": directory_bytes(teacher_dir),
        "teacher_dense_cache_bytes_diagnostic": estimates["teacher_dense_cache_bytes_diagnostic"],
        "uses_dense_all_node_teacher_cache": False,
        "uses_dense_teacher_cache_in_ram": False,
        "uses_teacher_probs_as_input": False,
        "uses_teacher_probs_as_soft_targets": True,
        "teacher_topk_build_mode": "streaming_logits",
        "teacher_train_time": float(teacher_train_time),
        "teacher_infer_time": float(time.perf_counter() - started),
        "valid_acc": "",
        "accuracy": "",
        "macro_f1": "",
        "created_at": utc_now(),
    }
    manifest["teacher_cache_id"] = stable_hash(
        {
            "mode": mode,
            "rows": target_size,
            "classes": num_classes,
            "build": "streaming_logits",
            "prior": prior.round(6).tolist(),
        }
    )
    write_json(teacher_dir / "teacher_cache_manifest.json", manifest)
    write_json(teacher_dir / "teacher_config.json", {"teacher": "prototype_streaming_logits", "mode": mode, "chunk_size": int(chunk_size)})
    write_json(teacher_dir / "teacher_metrics.json", {key: manifest[key] for key in ("valid_acc", "accuracy", "macro_f1", "teacher_train_time", "teacher_infer_time")})
    return manifest


def load_teacher_topk_cache(cache_root: str | Path) -> Papers100MTopKTeacherCache:
    root = Path(cache_root) / "teacher"
    manifest = read_json(root / "teacher_cache_manifest.json")
    shape = (int(manifest["target_universe_size"]), int(manifest["topk"]))
    return Papers100MTopKTeacherCache(
        root=root,
        manifest=manifest,
        topk_class_ids=np.memmap(root / manifest["topk_class_ids_path"], mode="r", dtype=np.uint16, shape=shape),
        topk_probs=np.memmap(root / manifest["topk_probs_path"], mode="r", dtype=np.float16, shape=shape),
        tail_mass=np.memmap(root / manifest["tail_mass_path"], mode="r", dtype=np.float16, shape=(shape[0],)),
        tail_prior=np.load(root / manifest["tail_prior_path"]),
        entropy=np.memmap(root / manifest["entropy_path"], mode="r", dtype=np.float16, shape=(shape[0],)),
        margin=np.memmap(root / manifest["margin_path"], mode="r", dtype=np.float16, shape=(shape[0],)),
    )


def evaluate_teacher_topk_cache(cache_root: str | Path) -> dict[str, Any]:
    cache_root = Path(cache_root)
    root_manifest = read_json(cache_root / "manifest.json")
    teacher = load_teacher_topk_cache(cache_root)
    target_size = int(root_manifest["target_universe_size"])
    num_nodes = int(root_manifest["num_nodes"])
    num_classes = int(root_manifest["num_classes"])
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(num_nodes,))

    def evaluate(split: str) -> dict[str, Any]:
        size = int(root_manifest[f"{split}_size"])
        rows = np.asarray(np.memmap(cache_root / "raw" / f"{split}_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(size,)), dtype=np.int64)
        node_ids = np.asarray(target_idx[rows], dtype=np.int64)
        split_labels = np.asarray(labels[node_ids], dtype=np.int64)
        pred = np.asarray(teacher.topk_class_ids[rows, 0], dtype=np.int64)
        return _classification_metrics(pred, split_labels, num_classes=num_classes)

    valid = evaluate("valid")
    test = evaluate("test")
    metrics = {
        "valid_acc": valid["accuracy"],
        "valid_macro_f1": valid["macro_f1"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "predicted_classes": test["predicted_classes"],
    }
    manifest = dict(teacher.manifest)
    manifest.update(metrics)
    write_json(cache_root / "teacher" / "teacher_cache_manifest.json", manifest)
    write_json(
        cache_root / "teacher" / "teacher_metrics.json",
        {key: manifest.get(key, "") for key in ("valid_acc", "valid_macro_f1", "accuracy", "macro_f1", "predicted_classes", "teacher_train_time", "teacher_infer_time")},
    )
    return metrics


def train_or_load_teacher(cache_root: str | Path, *, mode: str = "topk8_tail", force: bool = False) -> dict[str, Any]:
    cache_root = Path(cache_root)
    manifest_path = cache_root / "teacher" / "teacher_cache_manifest.json"
    if manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        if manifest.get("valid_acc", "") == "" or manifest.get("accuracy", "") == "":
            manifest.update(evaluate_teacher_topk_cache(cache_root))
        return read_json(manifest_path)
    root_manifest = read_json(cache_root / "manifest.json")
    sft_manifest = read_json(cache_root / "sft" / "sft_manifest.json")
    train_started = time.perf_counter()
    target_size = int(root_manifest["target_universe_size"])
    num_classes = int(root_manifest["num_classes"])
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(root_manifest["num_nodes"]),))
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    x0 = np.memmap(cache_root / "sft" / "X0_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, int(root_manifest["feature_dim"])))
    train_local = np.memmap(cache_root / "raw" / "train_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(root_manifest["train_size"]),))
    train_rows = np.asarray(train_local, dtype=np.int64)
    train_labels = np.asarray(labels[np.asarray(target_idx[train_rows], dtype=np.int64)], dtype=np.int64)
    prototypes = np.zeros((num_classes, x0.shape[1]), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for cls in range(num_classes):
        mask = train_labels == cls
        if np.any(mask):
            prototypes[cls] = np.asarray(x0[train_rows[mask]], dtype=np.float32).mean(axis=0)
            counts[cls] = float(mask.sum())
    if not np.any(counts):
        prototypes[:] = 0.0
    teacher_train_time = float(time.perf_counter() - train_started)
    class_bias = np.log(np.maximum(counts, 1.0)).astype(np.float32) * 0.05
    manifest = write_teacher_topk_cache_from_prototypes(
        cache_root,
        x0,
        prototypes,
        class_bias=class_bias,
        mode=mode,
        teacher_train_time=teacher_train_time,
    )
    manifest["sft_cache_id"] = sft_manifest.get("sft_cache_id", "")
    manifest.update(evaluate_teacher_topk_cache(cache_root))
    write_json(cache_root / "teacher" / "teacher_cache_manifest.json", manifest)
    return manifest
