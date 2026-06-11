from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.arxiv_cns_actual import MissingBaseLogitsError, load_base_logit_cache, run_t30_cns_grid
from shadow_hgc.sft.arxiv_cns_actual_v2 import cns_grid_plan, find_base_logits, is_historical_lad_predictor
from shadow_hgc.sft.t31_contract import ARXIV_NUM_CLASSES, T31_REQUIRED_FIELDS, apply_t31_promotion_guard, make_t31_row


def build_arxiv_cns_server_command() -> str:
    return (
        "python scripts/run_t31_arxiv_actual_cns.py --device cuda --base-predictors raw_x_mlp "
        "mlp_on_sft a4_sagn_lite_v4 sagn_lite_v5 gamlp_lite_v5 --train-base-logits-if-missing "
        "--enable-cns --correction-alphas 0.2 0.4 0.6 0.8 0.95 --smoothing-alphas 0.2 0.4 0.6 0.8 0.95 "
        "--correction-steps 10 20 50 100 --smoothing-steps 10 20 50 100 --autoscale on off "
        "--graph-directions cite_ref cited_by undirected_sym --hidden-dims 512 768 1024 --epochs 300 --seed 42 --run-long"
    )


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _load_gzip_ints(path: Path) -> torch.Tensor:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        arr = np.loadtxt(handle, delimiter=",", dtype=np.int64)
    return torch.from_numpy(np.atleast_1d(arr).astype(np.int64, copy=False)).to(torch.long)


def _load_gzip_float_matrix(path: Path) -> torch.Tensor:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        arr = np.loadtxt(handle, delimiter=",", dtype=np.float32)
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def _load_arxiv_arrays(dataset_root: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(dataset_root)
    labels = _load_gzip_ints(root / "raw" / "node-label.csv.gz")
    train = _load_gzip_ints(root / "split" / "time" / "train.csv.gz")
    valid = _load_gzip_ints(root / "split" / "time" / "valid.csv.gz")
    test = _load_gzip_ints(root / "split" / "time" / "test.csv.gz")
    features = _load_gzip_float_matrix(root / "raw" / "node-feat.csv.gz")
    with gzip.open(root / "raw" / "edge.csv.gz", "rt", encoding="utf-8") as handle:
        edge = np.loadtxt(handle, delimiter=",", dtype=np.int64)
    edge_index = torch.from_numpy(np.asarray(edge, dtype=np.int64).T).to(torch.long)
    return features, labels, train, valid, test, edge_index


class RawMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _metrics(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, num_classes: int) -> dict[str, Any]:
    pred = logits[rows].argmax(dim=1).cpu()
    y = labels[rows].cpu()
    confusion = torch.bincount((y * int(num_classes) + pred).clamp_min(0), minlength=int(num_classes) * int(num_classes)).view(int(num_classes), int(num_classes))
    tp = torch.diag(confusion).to(torch.float64)
    fp = confusion.sum(dim=0).to(torch.float64) - tp
    fn = confusion.sum(dim=1).to(torch.float64) - tp
    macro = float((2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-12)).mean().item())
    return {
        "accuracy": float((pred == y).float().mean().item()),
        "macro_f1": macro,
        "predicted_classes": int(pred.unique().numel()),
    }


def train_raw_x_mlp_logits(args: argparse.Namespace, out_path: Path) -> Path:
    started = time.perf_counter()
    torch.manual_seed(int(_arg(args, "seed", 42)))
    device_arg = str(_arg(args, "device", "cuda"))
    device = torch.device(device_arg if device_arg == "cpu" or torch.cuda.is_available() else "cpu")
    features, labels, train, valid, test, _edge_index = _load_arxiv_arrays(_arg(args, "dataset_root", "dataset/ogbn_arxiv"))
    mean = features[train].mean(dim=0, keepdim=True)
    std = features[train].std(dim=0, keepdim=True).clamp_min(1e-6)
    x = ((features - mean) / std).to(device)
    y = labels.to(device)
    hidden_dim = int(list(_arg(args, "hidden_dims", [512]))[0])
    model = RawMLP(x.shape[1], hidden_dim, ARXIV_NUM_CLASSES, dropout=0.35).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    batch_size = int(_arg(args, "batch_size", 8192))
    epochs = int(_arg(args, "epochs", 300))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_valid = -1.0
    generator = torch.Generator(device="cpu").manual_seed(int(_arg(args, "seed", 42)))
    train_cpu = train.cpu()
    for epoch in range(epochs):
        model.train()
        order = train_cpu[torch.randperm(train_cpu.numel(), generator=generator)]
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size].to(device)
            logits = model(x[batch])
            loss = F.cross_entropy(logits, y[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch == epochs - 1 or (epoch + 1) % max(1, min(20, epochs)) == 0:
            model.eval()
            with torch.no_grad():
                valid_logits = model(x[valid.to(device)])
                valid_acc = float((valid_logits.argmax(dim=1).cpu() == labels[valid]).float().mean().item())
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 32768):
            all_logits.append(model(x[start : start + 32768]).detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    valid_metrics = _metrics(logits, labels, valid, ARXIV_NUM_CLASSES)
    test_metrics = _metrics(logits, labels, test, ARXIV_NUM_CLASSES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logits": logits,
        "metadata": {
            "dataset": "ogbn-arxiv",
            "base_predictor": "raw_x_mlp",
            "seed": int(_arg(args, "seed", 42)),
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "num_nodes": int(logits.shape[0]),
            "num_classes": int(logits.shape[1]),
            "valid_acc": valid_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "predicted_classes": test_metrics["predicted_classes"],
            "training_config": {"lr": 0.003, "weight_decay": 1e-4, "dropout": 0.35, "batch_size": batch_size},
            "training_time": float(time.perf_counter() - started),
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
        },
    }
    torch.save(payload, out_path)
    return out_path


def _blocked_row(args: argparse.Namespace, predictor: str, reason: str, notes: str = "") -> dict[str, Any]:
    return make_t31_row(
        dataset="ogbn-arxiv",
        method=f"arxiv_{predictor}_cns_actual",
        seed=int(_arg(args, "seed", 42)),
        status="blocked",
        failure_reason=reason,
        promotion_track="safe_main",
        promotion_status="not_promoted",
        uses_cns_postprocess=True,
        uses_valid_labels_for_hyperparam_selection=True,
        base_predictor=predictor,
        next_action=build_arxiv_cns_server_command(),
        notes=notes,
    )


def build_arxiv_cns_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    predictors = list(_arg(args, "base_predictors", ["raw_x_mlp"]))
    base_dir = Path(_arg(args, "base_logits_dir", "experiments/logits/t31_arxiv"))
    plan = cns_grid_plan(
        correction_alphas=list(_arg(args, "correction_alphas", [0.2])),
        smoothing_alphas=list(_arg(args, "smoothing_alphas", [0.4])),
        correction_steps=list(_arg(args, "correction_steps", [10])),
        smoothing_steps=list(_arg(args, "smoothing_steps", [10])),
        autoscale=list(_arg(args, "autoscale", ["off"])),
        graph_directions=list(_arg(args, "graph_directions", ["cite_ref"])),
    )
    best_plan = plan[0] if plan else {"correction_alpha": "", "smoothing_alpha": "", "correction_steps": "", "smoothing_steps": "", "autoscale": "", "graph_direction": ""}
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        cache = find_base_logits(base_dir, predictor)
        if cache is None and bool(_arg(args, "train_base_logits_if_missing", False)) and predictor == "raw_x_mlp":
            cache = train_raw_x_mlp_logits(args, base_dir / "raw_x_mlp_logits.pt")
        if cache is None:
            rows.append(_blocked_row(args, predictor, "missing_base_logits"))
            continue
        try:
            loaded = load_base_logit_cache(cache)
        except MissingBaseLogitsError:
            rows.append(_blocked_row(args, predictor, "missing_base_logits"))
            continue
        diagnostic = is_historical_lad_predictor(predictor) or is_historical_lad_predictor(cache)
        if diagnostic:
            row = make_t31_row(
                dataset="ogbn-arxiv",
                method=f"arxiv_{predictor}_cns_actual",
                seed=int(_arg(args, "seed", 42)),
                status="blocked",
                failure_reason="historical_lad_diagnostic_not_main",
                promotion_track="safe_main",
                promotion_status="not_promoted",
                uses_cns_postprocess=True,
                uses_valid_labels_for_hyperparam_selection=True,
                base_predictor=predictor,
                base_accuracy=loaded.metadata.get("test_acc", loaded.metadata.get("accuracy", "")),
                base_macro_f1=loaded.metadata.get("macro_f1", ""),
                base_valid_acc=loaded.metadata.get("valid_acc", ""),
                best_correction_alpha=best_plan["correction_alpha"],
                best_smoothing_alpha=best_plan["smoothing_alpha"],
                best_correction_steps=best_plan["correction_steps"],
                best_smoothing_steps=best_plan["smoothing_steps"],
                autoscale=best_plan["autoscale"],
                graph_direction=best_plan["graph_direction"],
                source_table=str(cache),
                next_action=build_arxiv_cns_server_command(),
            )
            rows.append(apply_t31_promotion_guard(row))
            continue
        try:
            _features, labels, train, valid, test, edge_index = _load_arxiv_arrays(_arg(args, "dataset_root", "dataset/ogbn_arxiv"))
        except FileNotFoundError:
            rows.append(_blocked_row(args, predictor, "missing_arxiv_dataset"))
            continue
        result = run_t30_cns_grid(
            cache=loaded,
            labels=labels,
            train_idx=train,
            valid_idx=valid,
            test_idx=test,
            edge_index=edge_index,
            num_classes=ARXIV_NUM_CLASSES,
            correction_alphas=[float(v) for v in _arg(args, "correction_alphas", [0.4])],
            smoothing_alphas=[float(v) for v in _arg(args, "smoothing_alphas", [0.4])],
            correction_steps=[int(v) for v in _arg(args, "correction_steps", [20])],
            smoothing_steps=[int(v) for v in _arg(args, "smoothing_steps", [20])],
        )
        best = result.best_row
        row = make_t31_row(
            dataset="ogbn-arxiv",
            method=f"arxiv_{predictor}_cns_actual",
            seed=int(_arg(args, "seed", 42)),
            status="completed_long",
            failure_reason="",
            promotion_track="safe_main",
            promotion_status="not_promoted",
            uses_cns_postprocess=True,
            uses_valid_labels_for_hyperparam_selection=True,
            base_predictor=predictor,
            accuracy=best["accuracy"],
            macro_f1=best["macro_f1"],
            valid_acc=best["valid_acc"],
            predicted_classes=best["predicted_classes"],
            base_accuracy=loaded.metadata.get("test_acc", ""),
            base_macro_f1=loaded.metadata.get("macro_f1", ""),
            base_valid_acc=loaded.metadata.get("valid_acc", ""),
            cns_accuracy=best["accuracy"],
            cns_macro_f1=best["macro_f1"],
            cns_valid_acc=best["valid_acc"],
            best_correction_alpha=best["cns_correction_alpha"],
            best_smoothing_alpha=best["cns_smoothing_alpha"],
            best_correction_steps=best["cns_correction_steps"],
            best_smoothing_steps=best["cns_smoothing_steps"],
            autoscale=best_plan["autoscale"],
            graph_direction=best_plan["graph_direction"],
            source_table=str(cache),
            next_action=build_arxiv_cns_server_command(),
        )
        rows.append(apply_t31_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_cns_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t31_arxiv_actual_cns_seed42.csv"), rows, T31_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t31_arxiv_actual_cns_notes.md"),
        [
            "# T31 Arxiv Actual C&S",
            "",
            *markdown_table(rows, ["method", "base_predictor", "status", "failure_reason", "cns_accuracy", "cns_macro_f1", "graph_direction", "source_table"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_arxiv_cns_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T31 ogbn-arxiv actual C&S.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-predictors", nargs="+", default=["raw_x_mlp", "mlp_on_sft", "a4_sagn_lite_v4", "sagn_lite_v5", "gamlp_lite_v5"])
    parser.add_argument("--base-logits-dir", default="experiments/logits/t31_arxiv")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--train-base-logits-if-missing", action="store_true")
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--correction-alphas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--correction-steps", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--autoscale", nargs="+", default=["on", "off"])
    parser.add_argument("--graph-directions", nargs="+", default=["cite_ref", "cited_by", "undirected_sym"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t31_arxiv_actual_cns_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_arxiv_actual_cns_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
