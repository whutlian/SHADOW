from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.arxiv_actual_cns import MissingBaseLogitsError, require_base_logits, run_actual_cns_grid
from shadow_hgc.sft.t29_contract import ARXIV_NUM_CLASSES, T29_REQUIRED_FIELDS, make_t29_row


BASE_METHODS = {
    "raw_x_mlp": "arxiv_raw_mlp_cns_repro",
    "mlp_on_sft": "arxiv_sft_mlp_cns_actual",
    "sagn_lite_v5": "arxiv_sagn_sft_cns_actual",
    "gamlp_lite_v5": "arxiv_gamlp_sft_cns_actual",
    "time_aware_sft_head": "arxiv_sft_time_cns_actual",
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_arxiv_cns_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_arxiv_cns_actual.py --device cuda "
        "--base-predictors raw_x_mlp mlp_on_sft sagn_lite_v5 gamlp_lite_v5 "
        "--enable-cns --correction-alphas 0.2 0.4 0.6 0.8 0.95 "
        "--smoothing-alphas 0.2 0.4 0.6 0.8 0.95 "
        "--correction-steps 10 20 50 --smoothing-steps 10 20 50 "
        "--hidden-dims 512 768 1024 --epochs 300 "
        f"--seed {int(seed)} --run-long"
    )


def _logits_path(base_logits_dir: str | Path, predictor: str) -> Path:
    root = Path(base_logits_dir)
    for suffix in (".npy", ".pt"):
        path = root / f"{predictor}_logits{suffix}"
        if path.exists():
            return path
    return root / f"{predictor}_logits.npy"


def _load_gzip_ints(path: Path) -> torch.Tensor:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        arr = np.loadtxt(handle, delimiter=",", dtype=np.int64)
    return torch.from_numpy(np.atleast_1d(arr).astype(np.int64, copy=False)).to(torch.long)


def _load_arxiv_arrays(dataset_root: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(dataset_root)
    labels = _load_gzip_ints(root / "raw" / "node-label.csv.gz")
    train = _load_gzip_ints(root / "split" / "time" / "train.csv.gz")
    valid = _load_gzip_ints(root / "split" / "time" / "valid.csv.gz")
    test = _load_gzip_ints(root / "split" / "time" / "test.csv.gz")
    with gzip.open(root / "raw" / "edge.csv.gz", "rt", encoding="utf-8") as handle:
        edge = np.loadtxt(handle, delimiter=",", dtype=np.int64)
    edge_index = torch.from_numpy(np.asarray(edge, dtype=np.int64).T).to(torch.long)
    return labels, train, valid, test, edge_index


def _blocked_row(method: str, predictor: str, seed: int, reason: str, notes: str) -> dict[str, Any]:
    return make_t29_row(
        dataset="ogbn-arxiv",
        method=method,
        seed=seed,
        status="blocked",
        promotion_status="not_promoted",
        promotion_track="safe_mainline",
        failure_reason=reason,
        notes=notes,
        teacher_method=predictor,
        extra={"uses_cns_postprocess": True},
    )


def build_arxiv_cns_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    predictors = [str(v) for v in _arg(args, "base_predictors", ["raw_x_mlp"])]
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        method = BASE_METHODS.get(predictor, f"arxiv_{predictor}_cns_actual")
        path = _logits_path(_arg(args, "base_logits_dir", "experiments/logits/t29_arxiv"), predictor)
        try:
            logits = require_base_logits(path)
        except MissingBaseLogitsError:
            rows.append(
                _blocked_row(
                    method,
                    predictor,
                    seed,
                    "missing_base_logits",
                    f"Base logits were not found at {path}; run the actual base predictor first.",
                )
            )
            continue
        labels, train, valid, test, edge_index = _load_arxiv_arrays(_arg(args, "dataset_root", "dataset/ogbn_arxiv"))
        result = run_actual_cns_grid(
            logits=logits,
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
        rows.append(
            make_t29_row(
                dataset="ogbn-arxiv",
                method=method,
                seed=seed,
                accuracy=best["accuracy"],
                macro_f1=best["macro_f1"],
                valid_acc=best["valid_acc"],
                predicted_classes=best["predicted_classes"],
                teacher_accuracy=best["accuracy"],
                teacher_valid_acc=best["valid_acc"],
                teacher_macro_f1=best["macro_f1"],
                teacher_method=predictor,
                status="completed_real",
                promotion_status="not_promoted",
                promotion_track="safe_mainline",
                notes="Actual C&S run selected hyperparameters on validation accuracy only.",
                extra={
                    "uses_cns_postprocess": True,
                    "uses_valid_labels_as_input": False,
                    "uses_test_labels_as_input": False,
                    "source_table": str(path),
                },
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_cns_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_arxiv_cns_actual_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_arxiv_cns_actual_summary.md"),
        [
            "# T29 Arxiv Actual C&S",
            "",
            "- Rows require real base logits. Missing logits are blocked, not smoke-completed.",
            "- Validation labels are used only for hyperparameter selection; test labels are evaluation only.",
            "",
            *markdown_table(rows, ["method", "status", "accuracy", "macro_f1", "valid_acc", "teacher_method", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_arxiv_cns_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 actual arxiv C&S from real base logits.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-predictors", nargs="+", default=["raw_x_mlp", "mlp_on_sft", "sagn_lite_v5", "gamlp_lite_v5"])
    parser.add_argument("--base-logits-dir", default="experiments/logits/t29_arxiv")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--correction-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--correction-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_arxiv_cns_actual_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_arxiv_cns_actual_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
