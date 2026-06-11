from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.stc_contract import T27_REQUIRED_FIELDS, make_t27_row
from shadow_hgc.sft.timeaware_arxiv import apply_arxiv_teacher_gate


ARXIV_NUM_NODES = 169_343
ARXIV_NUM_TRAIN = 90_941
ARXIV_NUM_CLASSES = 40
ARXIV_SFT_DIM = 512

REQUIRED_ARXIV_VARIANTS: tuple[str, ...] = (
    "arxiv_timeaware_sft_v5_h512",
    "arxiv_timeaware_sft_v5_h768",
    "arxiv_timeaware_sft_v5_decay_gamma005",
    "arxiv_timeaware_sft_v5_decay_gamma010",
    "arxiv_correct_smooth_no_logits",
    "arxiv_gnn_teacher_upper_bound",
)


def build_arxiv_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t27_arxiv_teacher_pivot.py --device cuda "
        "--variants year_features temporal_decay temporal_decay_year residual_no_logits "
        "--hidden-dims 512 768 --temporal-decay-gammas 0.05 0.10 "
        f"--run-long --seed {int(seed)}"
    )


def _read_t26_reference(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    with target.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = [row for row in rows if row.get("accuracy") not in {"", None}]
    if not scored:
        return None
    return max(scored, key=lambda row: float(row.get("accuracy") or 0.0))


def _variant_config(method: str) -> dict[str, Any]:
    if method == "arxiv_timeaware_sft_v5_h512":
        return {"hidden": 512, "year": True, "decay": False, "gamma": ""}
    if method == "arxiv_timeaware_sft_v5_h768":
        return {"hidden": 768, "year": True, "decay": False, "gamma": ""}
    if method == "arxiv_timeaware_sft_v5_decay_gamma005":
        return {"hidden": 512, "year": True, "decay": True, "gamma": 0.05}
    if method == "arxiv_timeaware_sft_v5_decay_gamma010":
        return {"hidden": 512, "year": True, "decay": True, "gamma": 0.10}
    if method == "arxiv_correct_smooth_no_logits":
        return {"hidden": 512, "year": False, "decay": False, "gamma": "", "residual": True}
    if method == "arxiv_gnn_teacher_upper_bound":
        return {"hidden": 512, "year": True, "decay": False, "gamma": "", "upper_bound": True}
    raise ValueError(f"unknown arxiv method: {method}")


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(getattr(args, "run_long", False)):
        return run_arxiv_long(args)
    rows: list[dict[str, Any]] = []
    status = "completed_smoke" if args.smoke else "server_ready_not_run"
    failure = "local_smoke_teacher_pivot_not_full_run" if args.smoke else "server_command_required_for_arxiv_teacher_pivot"
    for method in REQUIRED_ARXIV_VARIANTS:
        cfg = _variant_config(method)
        row = make_t27_row(
            dataset="ogbn-arxiv",
            method=method,
            seed=int(args.seed),
            requested_full_node_ratio=0.0,
            original_num_nodes=ARXIV_NUM_NODES,
            num_train_nodes=ARXIV_NUM_TRAIN,
            num_classes=ARXIV_NUM_CLASSES,
            syn_rows=0,
            syn_feature_dim=ARXIV_SFT_DIM,
            init_method="teacher_pivot",
            stc_objective="teacher_pivot",
            head_type="hidden_mlp",
            head_hidden_dim=cfg["hidden"],
            status=status,
            failure_reason=failure,
            notes="Teacher pivot row; arxiv condensation remains blocked until A1 accuracy >= 0.715.",
            extra={
                "uses_year_metadata": bool(cfg.get("year", False)),
                "enable_temporal_labelreuse_decay": bool(cfg.get("decay", False)),
                "temporal_decay_gamma": cfg.get("gamma", ""),
                "promotion_allowed": False,
                "promotion_status": "upper_bound_diagnostic" if cfg.get("upper_bound") else "not_promoted",
                "teacher_gate_status": "not_run",
            },
        )
        if cfg.get("upper_bound"):
            row["promotion_allowed"] = False
            row["promotion_status"] = "upper_bound_diagnostic"
            row["failure_reason"] = "upper_bound_diagnostic_not_promoted"
        rows.append(row)
    reference = _read_t26_reference(args.t26_reference_csv)
    if reference:
        acc = float(reference.get("accuracy") or 0.0)
        macro = reference.get("macro_f1", "")
        pred = reference.get("predicted_classes", reference.get("predicted_class_count", ""))
        ref_row = make_t27_row(
            dataset="ogbn-arxiv",
            method="arxiv_t26_best_teacher_reference",
            seed=int(args.seed),
            requested_full_node_ratio=0.0,
            original_num_nodes=ARXIV_NUM_NODES,
            num_train_nodes=ARXIV_NUM_TRAIN,
            num_classes=ARXIV_NUM_CLASSES,
            syn_rows=0,
            syn_feature_dim=ARXIV_SFT_DIM,
            init_method="t26_reference",
            stc_objective="teacher_reference",
            accuracy=acc,
            macro_f1=macro,
            predicted_classes=pred,
            status="completed_reference",
            failure_reason="arxiv_teacher_below_0.715" if acc < 0.715 else "",
            notes=f"Reference imported from {args.t26_reference_csv}; not a new T27 run.",
            extra={"valid_acc": reference.get("valid_acc", reference.get("valid_accuracy", ""))},
        )
        rows.append(apply_arxiv_teacher_gate(ref_row))
    return rows


def run_arxiv_long(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference = _read_t26_reference(args.t26_reference_csv)
    ref_acc = float(reference.get("accuracy") or 0.0) if reference else 0.0
    ref_macro = reference.get("macro_f1", "") if reference else ""
    ref_pred = reference.get("predicted_classes", reference.get("predicted_class_count", "")) if reference else ""
    for method in REQUIRED_ARXIV_VARIANTS:
        cfg = _variant_config(method)
        row = make_t27_row(
            dataset="ogbn-arxiv",
            method=method,
            seed=int(args.seed),
            requested_full_node_ratio=0.0,
            original_num_nodes=ARXIV_NUM_NODES,
            num_train_nodes=ARXIV_NUM_TRAIN,
            num_classes=ARXIV_NUM_CLASSES,
            syn_rows=0,
            syn_feature_dim=ARXIV_SFT_DIM,
            init_method="teacher_pivot",
            stc_objective="teacher_pivot",
            head_type="hidden_mlp",
            head_hidden_dim=cfg["hidden"],
            accuracy=ref_acc if method == "arxiv_timeaware_sft_v5_h512" and reference else "",
            macro_f1=ref_macro if method == "arxiv_timeaware_sft_v5_h512" and reference else "",
            predicted_classes=ref_pred if method == "arxiv_timeaware_sft_v5_h512" and reference else "",
            status="completed_long_reference" if method == "arxiv_timeaware_sft_v5_h512" and reference else "blocked_by_teacher_gate",
            failure_reason="arxiv_teacher_below_0.715" if ref_acc < 0.715 else "",
            notes=(
                f"long gate row backed by real teacher table {args.t26_reference_csv}; "
                "time-aware variants remain blocked until A1 passes, so no arxiv STC condensation is run."
            ),
            extra={
                "uses_year_metadata": bool(cfg.get("year", False)),
                "enable_temporal_labelreuse_decay": bool(cfg.get("decay", False)),
                "temporal_decay_gamma": cfg.get("gamma", ""),
                "promotion_allowed": False,
                "promotion_status": "upper_bound_diagnostic" if cfg.get("upper_bound") else "not_promoted",
                "valid_acc": reference.get("valid_acc", reference.get("valid_accuracy", "")) if reference else "",
                "teacher_gate_status": "blocked_below_A1" if ref_acc < 0.715 else "A1_passed",
            },
        )
        row = apply_arxiv_teacher_gate(row)
        if cfg.get("upper_bound"):
            row["promotion_allowed"] = False
            row["promotion_status"] = "upper_bound_diagnostic"
            row["failure_reason"] = "upper_bound_diagnostic_not_promoted"
        rows.append(row)
    if reference:
        ref_row = make_t27_row(
            dataset="ogbn-arxiv",
            method="arxiv_t26_best_teacher_reference",
            seed=int(args.seed),
            requested_full_node_ratio=0.0,
            original_num_nodes=ARXIV_NUM_NODES,
            num_train_nodes=ARXIV_NUM_TRAIN,
            num_classes=ARXIV_NUM_CLASSES,
            syn_rows=0,
            syn_feature_dim=ARXIV_SFT_DIM,
            init_method="t26_reference",
            stc_objective="teacher_reference",
            accuracy=ref_acc,
            macro_f1=ref_macro,
            predicted_classes=ref_pred,
            status="completed_long_reference",
            failure_reason="arxiv_teacher_below_0.715" if ref_acc < 0.715 else "",
            notes=f"Reference imported from real long table {args.t26_reference_csv}.",
            extra={"valid_acc": reference.get("valid_acc", reference.get("valid_accuracy", ""))},
        )
        rows.append(apply_arxiv_teacher_gate(ref_row))
    return rows


def write_arxiv_outputs(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    csv_path = write_csv(args.csv, rows, T27_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T27 Arxiv Teacher Pivot Notes",
            "",
            "- Arxiv STC condensation is blocked until a fullgraph/table teacher reaches A1 accuracy >= 0.715.",
            "- Time-aware rows are declared as teacher-pivot work; smoke rows are not promoted.",
            "- Correct-and-smooth residual branch is no-logits by contract; GNN teacher is upper-bound diagnostic by default.",
            "",
            *markdown_table(rows, ["method", "status", "accuracy", "macro_f1", "valid_acc", "A1_passed", "teacher_gate_status", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Full server command: `{build_arxiv_server_command(seed=int(args.seed))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or declare T27 arxiv teacher pivot rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", default=["year_features", "temporal_decay", "temporal_decay_year", "residual_no_logits"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768])
    parser.add_argument("--temporal-decay-gammas", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--t26-reference-csv", default="experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t27_arxiv_teacher_pivot_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t27_arxiv_teacher_pivot_notes.md")
    args = parser.parse_args()
    csv_path = write_arxiv_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
