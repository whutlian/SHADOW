from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.models.sft_table_teacher import SFTTableTeacherV2
from shadow_hgc.train.train_sft_teacher import sft_loss


def _block_key(name: str) -> str:
    if name == "X0":
        return "self"
    return name.lower()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class LazyMemmapBlockStore:
    root: Path
    arrays: dict[str, np.memmap]
    block_dims: dict[str, int]
    stats: dict[str, dict[str, Any]]
    num_rows: int
    max_batch_materialized_bytes: int = 0

    @classmethod
    def from_manifest(cls, root: str | Path, manifest: dict[str, Any]) -> "LazyMemmapBlockStore":
        base = Path(root)
        arrays: dict[str, np.memmap] = {}
        dims: dict[str, int] = {}
        stats: dict[str, dict[str, Any]] = {}
        num_rows: int | None = None
        for block in manifest.get("blocks", []):
            name = str(block["name"])
            key = _block_key(name)
            shape = tuple(int(value) for value in block["shape"])
            dtype = np.dtype(str(block["dtype"]))
            path = base / str(block["path"])
            arrays[key] = np.memmap(path, mode="r", dtype=dtype, shape=shape)
            dims[key] = int(shape[1])
            num_rows = int(shape[0]) if num_rows is None else min(num_rows, int(shape[0]))
            stats_path = base / f"block_{name}_stats.json"
            raw_stats = _read_json(stats_path)
            stats[key] = {
                "source": raw_stats.get("fit_scope", raw_stats.get("source", "train_target_rows")),
                "frozen": bool(raw_stats.get("frozen", True)),
                "fit_rows": raw_stats.get("fit_rows", []),
                "mean": raw_stats.get("mean", [0.0] * int(shape[1])),
                "std": raw_stats.get("std", [1.0] * int(shape[1])),
            }
        if not arrays or num_rows is None:
            raise ValueError("manifest does not contain any memmap blocks")
        return cls(root=base, arrays=arrays, block_dims=dims, stats=stats, num_rows=int(num_rows))

    def fetch(self, rows: torch.Tensor | np.ndarray, *, device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        if isinstance(rows, torch.Tensor):
            row_index = rows.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            row_index = np.asarray(rows, dtype=np.int64)
        out: dict[str, torch.Tensor] = {}
        materialized = 0
        for key, array in self.arrays.items():
            values = np.asarray(array[row_index], dtype=np.float32)
            materialized += int(values.nbytes)
            tensor = torch.from_numpy(values)
            out[key] = tensor.to(device=device, dtype=dtype, non_blocking=False)
        self.max_batch_materialized_bytes = max(self.max_batch_materialized_bytes, materialized)
        return out


def load_manifest_block_store(manifest_dir: str | Path) -> LazyMemmapBlockStore:
    root = Path(manifest_dir)
    return LazyMemmapBlockStore.from_manifest(root, _read_json(root / "manifest.json"))


def load_ogb_labels_and_splits(dataset_root: str | Path, *, split_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(dataset_root)
    labels = torch.from_numpy(_load_gzip_ints(root / "raw" / "node-label.csv.gz")).to(torch.long)
    split_dir = root / "split" / str(split_name)
    train = torch.from_numpy(_load_gzip_ints(split_dir / "train.csv.gz")).to(torch.long)
    valid = torch.from_numpy(_load_gzip_ints(split_dir / "valid.csv.gz")).to(torch.long)
    test = torch.from_numpy(_load_gzip_ints(split_dir / "test.csv.gz")).to(torch.long)
    return labels, train, valid, test


def load_products_labels_and_splits(dataset_root: str | Path = "dataset/ogbn_products") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return load_ogb_labels_and_splits(dataset_root, split_name="sales_ranking")


def load_arxiv_labels_and_splits(dataset_root: str | Path = "dataset/ogbn_arxiv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return load_ogb_labels_and_splits(dataset_root, split_name="time")


def _load_gzip_ints(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, delimiter=",")
    return np.atleast_1d(values).astype(np.int64, copy=False)


def _load_block_stats_into_model(model: SFTTableTeacherV2, store: LazyMemmapBlockStore) -> None:
    for key, normalizer in model.normalizers.items():
        stats = store.stats[key]
        mean = torch.tensor(stats["mean"], dtype=normalizer.mean.dtype)
        std = torch.tensor(stats["std"], dtype=normalizer.std.dtype).clamp_min(1e-6)
        normalizer.mean.copy_(mean)
        normalizer.std.copy_(std)
        normalizer.fit_scope = str(stats.get("source", "train_target_rows"))
        normalizer.fit_rows = [int(value) for value in stats.get("fit_rows", [])]
        normalizer.fitted = True
        normalizer.frozen = True


@dataclass
class LazySFTTrainResult:
    summary: dict[str, Any]


def _iter_batches(rows: torch.Tensor, *, batch_size: int, shuffle: bool, seed: int) -> list[torch.Tensor]:
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
        rows = rows[torch.randperm(rows.numel(), generator=generator)]
    return [rows[start : start + int(batch_size)].contiguous() for start in range(0, int(rows.numel()), int(batch_size))]


def _metrics_from_counts(confusion: torch.Tensor, pred_hist: torch.Tensor) -> dict[str, Any]:
    correct = float(torch.diag(confusion).sum().item())
    total = float(confusion.sum().item())
    tp = torch.diag(confusion).to(torch.float64)
    fp = confusion.sum(dim=0).to(torch.float64) - tp
    fn = confusion.sum(dim=1).to(torch.float64) - tp
    denom = (2.0 * tp + fp + fn).clamp_min(1e-12)
    macro_f1 = float((2.0 * tp / denom).mean().item()) if confusion.numel() else 0.0
    probs = pred_hist.to(torch.float64) / pred_hist.sum().clamp_min(1).to(torch.float64)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item()) if pred_hist.numel() else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "predicted_class_count": int((pred_hist > 0).sum().item()),
        "prediction_entropy": entropy,
    }


def evaluate_lazy_sft(
    model: SFTTableTeacherV2,
    store: LazyMemmapBlockStore,
    labels: torch.Tensor,
    rows: torch.Tensor,
    *,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(int(num_classes), int(num_classes), dtype=torch.int64)
    pred_hist = torch.zeros(int(num_classes), dtype=torch.int64)
    with torch.no_grad():
        for batch_rows in _iter_batches(rows.to(torch.long), batch_size=batch_size, shuffle=False, seed=0):
            blocks = store.fetch(batch_rows, device=device)
            logits = model(blocks)
            pred = logits.argmax(dim=1).detach().cpu().to(torch.long)
            y = labels[batch_rows].to(torch.long).cpu()
            encoded = y.clamp_min(0) * int(num_classes) + pred.clamp_min(0)
            counts = torch.bincount(encoded, minlength=int(num_classes) * int(num_classes)).view(int(num_classes), int(num_classes))
            confusion += counts
            pred_hist += torch.bincount(pred.clamp_min(0), minlength=int(num_classes))
    return _metrics_from_counts(confusion, pred_hist)


def train_lazy_sft_from_memmap(
    *,
    manifest_dir: str | Path,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    test_rows: torch.Tensor,
    num_classes: int,
    device: str = "cuda",
    model_type: str = "gamlp_lite",
    hidden_dim: int = 128,
    dropout: float = 0.3,
    loss_type: str = "sqrt_weighted_ce",
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 16_384,
    eval_batch_size: int = 65_536,
    seed: int = 42,
    label_smoothing: float = 0.0,
) -> LazySFTTrainResult:
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    store = load_manifest_block_store(manifest_dir)
    model = SFTTableTeacherV2(
        store.block_dims,
        num_classes=int(num_classes),
        model_type=model_type,  # type: ignore[arg-type]
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(target_device)
    _load_block_stats_into_model(model, store)
    labels = labels.to(torch.long).cpu()
    train_rows = train_rows.to(torch.long).cpu()
    valid_rows = valid_rows.to(torch.long).cpu()
    test_rows = test_rows.to(torch.long).cpu()
    train_labels = labels[train_rows].to(torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_valid: dict[str, Any] | None = None
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_rows in _iter_batches(train_rows, batch_size=int(batch_size), shuffle=True, seed=int(seed) + epoch):
            opt.zero_grad(set_to_none=True)
            blocks = store.fetch(batch_rows, device=target_device)
            y = labels[batch_rows].to(device=target_device, dtype=torch.long)
            logits = model(blocks)
            loss = sft_loss(logits, y, loss_type=loss_type, train_labels=train_labels.to(target_device), label_smoothing=label_smoothing)
            loss.backward()
            opt.step()
            batch_size_seen = int(batch_rows.numel())
            total_loss += float(loss.detach().cpu().item()) * batch_size_seen
            seen += batch_size_seen
        valid = evaluate_lazy_sft(
            model,
            store,
            labels,
            valid_rows,
            num_classes=int(num_classes),
            batch_size=int(eval_batch_size),
            device=target_device,
        )
        valid["epoch"] = int(epoch)
        valid["train_loss"] = total_loss / max(1, seen)
        best_valid = valid
    test_started = time.perf_counter()
    test = evaluate_lazy_sft(
        model,
        store,
        labels,
        test_rows,
        num_classes=int(num_classes),
        batch_size=int(eval_batch_size),
        device=target_device,
    )
    inference_time = float(time.perf_counter() - test_started)
    summary = {
        "model_type": model_type,
        "loss_type": loss_type,
        "seed": int(seed),
        "epochs_ran": int(epochs),
        "batch_size": int(batch_size),
        "eval_batch_size": int(eval_batch_size),
        "device": str(target_device),
        "block_dims": dict(store.block_dims),
        "num_blocks": int(len(store.block_dims)),
        "num_rows": int(store.num_rows),
        "train_rows": int(train_rows.numel()),
        "valid_rows": int(valid_rows.numel()),
        "test_rows": int(test_rows.numel()),
        "valid": best_valid or {},
        "test": test,
        "training_time_s": float(time.perf_counter() - started),
        "inference_time_s": inference_time,
        "max_batch_materialized_bytes": int(store.max_batch_materialized_bytes),
        "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
        "uses_lazy_memmap": True,
        "loads_edge_index": False,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
        "uses_diffusion_legacy": False,
        **model.diagnostics(),
    }
    return LazySFTTrainResult(summary=summary)
