from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.metrics import macro_f1_score


FIELDS = [
    "dataset",
    "seed",
    "status",
    "num_nodes",
    "num_train",
    "num_valid",
    "num_test",
    "num_classes_from_dataset",
    "num_classes_from_labels",
    "output_dim",
    "label_min",
    "label_max",
    "label_shape",
    "feature_shape",
    "feature_dtype",
    "feature_nan_count",
    "feature_mean_std_sample",
    "train_mask_hash",
    "valid_mask_hash",
    "test_mask_hash",
    "first_10_train_ids",
    "first_10_train_labels",
    "first_10_valid_ids",
    "first_10_valid_labels",
    "predicted_class_count",
    "prediction_entropy",
    "train_acc",
    "valid_acc",
    "test_acc",
    "macro_f1",
    "weighted_f1",
    "uses_ogb_evaluator",
    "uses_bounded_edges",
    "reason",
]


def _hash_tensor(values: torch.Tensor) -> str:
    x = values.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def _squeeze_labels(labels: torch.Tensor) -> torch.Tensor:
    y = labels
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    if y.ndim != 1:
        raise ValueError("labels are not 1-D after squeeze")
    return y.to(torch.long)


def _check_no_overlap(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    overlap = torch.isin(a, b)
    if bool(overlap.any()):
        raise ValueError(f"{name_a}/{name_b} split overlap")


def validate_products_self_parity_inputs(
    *,
    num_nodes: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
    output_dim: int,
    num_classes_from_dataset: int,
    uses_ogb_evaluator: bool,
    uses_bounded_edges: bool,
) -> dict[str, Any]:
    y = _squeeze_labels(labels)
    if int(output_dim) != int(num_classes_from_dataset):
        raise ValueError("output_dim != num_classes_from_dataset")
    if int(features.shape[0]) != int(num_nodes):
        raise ValueError("feature rows != num_nodes")
    if features.ndim != 2 or int(features.shape[1]) == 0:
        raise ValueError("feature dim == 0")
    if int(y.min().item()) < 0:
        raise ValueError("label_min < 0")
    if int(y.max().item()) >= int(output_dim):
        raise ValueError("label_max >= output_dim")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("feature contains NaN or Inf")
    _check_no_overlap("train", train_idx, "valid", valid_idx)
    _check_no_overlap("train", train_idx, "test", test_idx)
    _check_no_overlap("valid", valid_idx, "test", test_idx)
    if not uses_ogb_evaluator:
        raise ValueError("OGB evaluator is not used")
    if uses_bounded_edges:
        raise ValueError("bounded_edges affects self-only path")
    sample = features[: min(4096, features.shape[0])].to(torch.float32)
    return {
        "num_nodes": int(num_nodes),
        "num_train": int(train_idx.numel()),
        "num_valid": int(valid_idx.numel()),
        "num_test": int(test_idx.numel()),
        "num_classes_from_dataset": int(num_classes_from_dataset),
        "num_classes_from_labels": int(y.max().item()) + 1,
        "output_dim": int(output_dim),
        "label_min": int(y.min().item()),
        "label_max": int(y.max().item()),
        "label_shape": [int(v) for v in y.shape],
        "feature_shape": [int(v) for v in features.shape],
        "feature_dtype": str(features.dtype).replace("torch.", ""),
        "feature_nan_count": int(torch.isnan(features).sum().item()),
        "feature_mean_std_sample": {
            "mean": float(sample.mean().item()),
            "std": float(sample.std(unbiased=False).item()),
        },
        "train_mask_hash": _hash_tensor(train_idx.to(torch.long)),
        "valid_mask_hash": _hash_tensor(valid_idx.to(torch.long)),
        "test_mask_hash": _hash_tensor(test_idx.to(torch.long)),
        "first_10_train_ids": [int(v) for v in train_idx[:10].tolist()],
        "first_10_train_labels": [int(v) for v in y[train_idx[:10]].tolist()],
        "first_10_valid_ids": [int(v) for v in valid_idx[:10].tolist()],
        "first_10_valid_labels": [int(v) for v in y[valid_idx[:10]].tolist()],
        "uses_ogb_evaluator": bool(uses_ogb_evaluator),
        "uses_bounded_edges": bool(uses_bounded_edges),
    }


class ProductsSelfMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, layers: int, dropout: float, batchnorm: bool) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        dim = int(in_dim)
        for _ in range(max(1, int(layers) - 1)):
            modules.append(torch.nn.Linear(dim, int(hidden_dim)))
            if batchnorm:
                modules.append(torch.nn.BatchNorm1d(int(hidden_dim)))
            modules.extend([torch.nn.ReLU(), torch.nn.Dropout(float(dropout))])
            dim = int(hidden_dim)
        modules.append(torch.nn.Linear(dim, int(out_dim)))
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _manual_weighted_f1(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> float:
    selected_pred = pred[idx]
    selected_labels = labels[idx]
    total = 0.0
    score = 0.0
    for class_id in range(num_classes):
        true_mask = selected_labels == class_id
        pred_mask = selected_pred == class_id
        support = float(true_mask.sum().item())
        if support == 0:
            continue
        tp = float((true_mask & pred_mask).sum().item())
        fp = float((~true_mask & pred_mask).sum().item())
        fn = float((true_mask & ~pred_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        score += support * f1
        total += support
    return score / total if total > 0 else 0.0


def _evaluate_split(evaluator, pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    result = evaluator.eval({"y_true": labels[idx].view(-1, 1), "y_pred": pred[idx].view(-1, 1)})
    return float(result["acc"])


def _predict_all(model: torch.nn.Module, features: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            batch = features[start : start + int(batch_size)].to(device)
            preds.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(preds, dim=0)


def _fit_train_standardizer(features: torch.Tensor, train_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = features[train_idx].to(torch.float32)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def run_products_self_parity(args) -> dict[str, Any]:
    try:
        from ogb.nodeproppred import Evaluator
    except ImportError as exc:
        raise RuntimeError("ogb is required for products self parity") from exc
    torch.manual_seed(int(args.seed))
    graph = load_ogb_node_property_dataset("ogbn-products", root=args.root, download=args.download)
    labels = graph.labels.to(torch.long)
    num_classes = int(labels.max().item()) + 1
    output_dim = num_classes
    diagnostics = validate_products_self_parity_inputs(
        num_nodes=graph.num_nodes[graph.target_type],
        features=graph.node_features[graph.target_type],
        labels=labels,
        train_idx=graph.train_idx,
        valid_idx=graph.val_idx,
        test_idx=graph.test_idx,
        output_dim=output_dim,
        num_classes_from_dataset=num_classes,
        uses_ogb_evaluator=True,
        uses_bounded_edges=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = graph.node_features[graph.target_type].to(torch.float32)
    y = labels
    if args.standardize_train_features:
        mean, std = _fit_train_standardizer(x, graph.train_idx)
        x = (x - mean) / std
    model = ProductsSelfMLP(
        x.shape[1],
        args.hidden_dim,
        output_dim,
        layers=args.layers,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = graph.train_idx.to(torch.long)
    started = time.perf_counter()
    for _ in range(int(args.epochs)):
        order = train_idx[torch.randperm(train_idx.numel())]
        model.train()
        for start in range(0, int(order.numel()), int(args.batch_size)):
            batch_idx = order[start : start + int(args.batch_size)]
            opt.zero_grad(set_to_none=True)
            logits = model(x[batch_idx].to(device))
            loss = F.cross_entropy(logits, y[batch_idx].to(device))
            loss.backward()
            opt.step()
    evaluator = Evaluator(name="ogbn-products")
    pred = _predict_all(model, x, batch_size=args.eval_batch_size, device=device)
    hist = torch.bincount(pred[graph.test_idx], minlength=num_classes).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item()) if hist.numel() else 0.0
    row = {
        "dataset": "ogbn-products",
        "seed": int(args.seed),
        "status": "completed",
        "reason": "",
        **diagnostics,
        "predicted_class_count": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
        "train_acc": _evaluate_split(evaluator, pred, y, graph.train_idx),
        "valid_acc": _evaluate_split(evaluator, pred, y, graph.val_idx),
        "test_acc": _evaluate_split(evaluator, pred, y, graph.test_idx),
        "macro_f1": macro_f1_score(pred[graph.test_idx], y[graph.test_idx], num_classes=num_classes),
        "weighted_f1": _manual_weighted_f1(pred, y, graph.test_idx, num_classes),
        "training_time_s": float(time.perf_counter() - started),
        "device": str(device),
        "epochs": int(args.epochs),
        "standardize_train_features": bool(args.standardize_train_features),
        "batchnorm": bool(args.batchnorm),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FIELDS)
    for key in sorted({key for row in rows for key in row}):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, row: dict[str, Any], csv_path: Path) -> None:
    status = row.get("status", "")
    test_acc = row.get("test_acc", "")
    passed = status == "completed" and test_acc != "" and float(test_acc) >= 0.50
    lines = [
        "# Products Self-Only Parity Seed 42",
        "",
        f"- Status: `{status}`",
        f"- Test accuracy: `{test_acc}`",
        f"- Acceptance >= 0.50: `{passed}`",
        f"- Uses OGB evaluator: `{row.get('uses_ogb_evaluator')}`",
        f"- Uses bounded edges: `{row.get('uses_bounded_edges')}`",
        f"- CSV: `{csv_path}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug ogbn-products self-only raw-feature MLP parity.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batchnorm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--standardize-train-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="experiments/tables/products_self_parity_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/products_self_parity_summary.md")
    parser.add_argument("--json-output", default="experiments/logs/products_self_parity_seed42/products_self_mlp_seed42.json")
    args = parser.parse_args()
    json_path = Path(args.json_output)
    try:
        row = run_products_self_parity(args)
    except Exception as exc:
        row = {
            "dataset": "ogbn-products",
            "seed": int(args.seed),
            "status": "blocked_by_data_path_bug",
            "reason": str(exc),
            "uses_ogb_evaluator": False,
            "uses_bounded_edges": False,
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    output = Path(args.output)
    _write_csv(output, [row])
    _write_report(Path(args.report), row, output)
    print(json.dumps({"status": row.get("status"), "test_acc": row.get("test_acc"), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
