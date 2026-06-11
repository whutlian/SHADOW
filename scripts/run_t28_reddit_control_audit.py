from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t28_contract import (
    REDDIT_NUM_NODES,
    REDDIT_STRUCTURE_FIELDS,
    apply_t28_promotion_guard,
    make_reddit_structure_row,
    reddit_gate_passed,
)


DEFAULT_RATIOS: tuple[float, ...] = (0.0005, 0.001, 0.0025, 0.005, 0.01)
DEFAULT_METHODS: tuple[str, ...] = (
    "current_sft_signature_random",
    "current_sft_signature_medoid",
    "current_sft_signature_kcenter",
    "sft_hnr_fdm_hybrid",
    "reddit_random_frozen_init",
)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_reddit_control_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t28_reddit_control_audit.py --device cuda "
        "--ratios 0.0005 0.001 0.0025 0.005 0.01 "
        "--methods current_sft_signature_random current_sft_signature_medoid current_sft_signature_kcenter "
        "sft_hnr_fdm_hybrid reddit_random_frozen_init "
        "--hidden-dims 128 256 512 --epochs 30 60 120 --seeds 1 2 3 4 5 42 --run-long"
    )


def _load_reference_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv(path):
            copied = dict(row)
            copied["_source_csv"] = str(path)
            rows.append(copied)
    return rows


def _method_aliases(method: str) -> set[str]:
    aliases = {method}
    if method == "current_sft_signature_random":
        aliases.add("SFT-signature random")
    elif method == "current_sft_signature_medoid":
        aliases.add("SFT-signature medoid")
    elif method == "current_sft_signature_kcenter":
        aliases.add("SFT-signature kcenter")
    return aliases


def _match_reference(references: list[dict[str, Any]], method: str, ratio: float, seed: int) -> dict[str, Any] | None:
    aliases = _method_aliases(method)
    candidates: list[dict[str, Any]] = []
    for row in references:
        if row.get("method") not in aliases:
            continue
        if row.get("seed") not in {"", None} and int(float(row.get("seed", 0))) != int(seed):
            continue
        if abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) > 1e-12:
            continue
        if row.get("accuracy") in {"", None}:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: fvalue(row.get("accuracy")))


def _selector_for(method: str) -> str:
    if method == "reddit_random_frozen_init":
        return "current_sft_signature_random"
    return method


def _control_row_from_reference(method: str, ratio: float, seed: int, ref: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    syn_rows = int(fvalue(ref.get("syn_rows", ref.get("target_prototypes", 0)), default=0.0))
    if syn_rows <= 0:
        syn_rows = max(1, int(round(float(ratio) * REDDIT_NUM_NODES)))
    row = make_reddit_structure_row(
        method=method,
        seed=seed,
        requested_full_node_ratio=float(ratio),
        target_prototypes=syn_rows,
        shadow_nodes=0,
        synthetic_rows=0,
        condensed_edges=0,
        prototype_selector=_selector_for(method),
        edge_builder="table_only",
        student_model="sagn_lite_v4_synthetic_table",
        accuracy=ref.get("accuracy", ""),
        macro_f1=ref.get("macro_f1", ""),
        predicted_classes=ref.get("predicted_classes", ref.get("predicted_class_count", "")),
        valid_acc=ref.get("valid_acc", ref.get("valid_accuracy", "")),
        status="completed_reference",
        promotion_status="not_promoted",
        failure_reason="control_reference_not_structure_promotion",
        notes=f"Imported control reference from {ref.get('_source_csv', '')}; smoke={bool(smoke)}.",
        extra={
            "precompute_time": ref.get("precompute_time", ""),
            "condensation_time": ref.get("total_time", ref.get("condensation_time", "")),
            "student_training_time": ref.get("final_training_time", ref.get("training_time", "")),
            "eval_time": ref.get("inference_time", ""),
            "peak_cpu_ram": ref.get("peak_cpu_ram", ""),
            "peak_gpu_ram": ref.get("peak_gpu_ram", ""),
            "cache_bytes": ref.get("cache_bytes", ""),
            "byte_compression": "",
        },
    )
    return apply_t28_promotion_guard(row, dataset_gate_passed=False)


def build_control_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(value) for value in _arg(args, "ratios", DEFAULT_RATIOS)]
    methods = [str(value) for value in _arg(args, "methods", DEFAULT_METHODS)]
    seeds = [int(value) for value in _arg(args, "seeds", [_arg(args, "seed", 42)])]
    smoke = bool(_arg(args, "smoke", False))
    references = _load_reference_rows(
        [
            _arg(args, "t27_reddit_csv", "experiments/tables/t27_stc_reddit_0p05_0p1_0p5_percent_seed42.csv"),
            _arg(args, "t25_reddit_csv", "experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv"),
            _arg(args, "t24_reddit_csv", "experiments/tables/t24_reddit_sft_condense_seed42.csv"),
        ]
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for ratio in ratios:
            for method in methods:
                ref = _match_reference(references, method, ratio, seed)
                if ref is not None:
                    rows.append(_control_row_from_reference(method, ratio, seed, ref, smoke=smoke))
                    continue
                total = max(1, int(round(float(ratio) * REDDIT_NUM_NODES)))
                rows.append(
                    make_reddit_structure_row(
                        method=method,
                        seed=seed,
                        requested_full_node_ratio=float(ratio),
                        target_prototypes=total,
                        edge_builder="table_only",
                        prototype_selector=_selector_for(method),
                        student_model="sagn_lite_v4_synthetic_table",
                        status="completed_smoke" if smoke else "server_ready_not_run",
                        promotion_status="not_promoted",
                        failure_reason="smoke_only_no_training" if smoke else "server_command_required_for_control_audit",
                        notes="Control-audit row; table-only baseline, no condensed graph edges.",
                    )
                )
    return rows


def write_control_outputs(args: argparse.Namespace) -> Path:
    rows = build_control_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t28_reddit_control_audit_seed_sweep.csv"), rows, REDDIT_STRUCTURE_FIELDS)
    t25 = [
        row
        for row in rows
        if row.get("method") == "sft_hnr_fdm_hybrid" and abs(fvalue(row.get("requested_full_node_ratio")) - 0.001) < 1e-12
    ]
    frozen = [
        row
        for row in rows
        if row.get("method") == "reddit_random_frozen_init" and abs(fvalue(row.get("requested_full_node_ratio")) - 0.001) < 1e-12
    ]
    ensure_report(
        _arg(args, "report", "experiments/summaries/t28_reddit_control_audit_summary.md"),
        [
            "# T28 Reddit Control Audit",
            "",
            "- Table-only controls are retained for reconciliation, not promoted as structure-aware rows.",
            "- The audit explicitly distinguishes T25 HNR-FDM references from T27 frozen-random rows.",
            "",
            "## 0.10% Reconciliation",
            *markdown_table(t25 + frozen, ["method", "status", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "failure_reason"]),
            "",
            "## All Rows",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "status", "accuracy", "macro_f1", "edge_builder", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_reddit_control_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T28 Reddit table-only control audit and reconciliation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[30, 60, 120])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--t27-reddit-csv", default="experiments/tables/t27_stc_reddit_0p05_0p1_0p5_percent_seed42.csv")
    parser.add_argument("--t25-reddit-csv", default="experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv")
    parser.add_argument("--t24-reddit-csv", default="experiments/tables/t24_reddit_sft_condense_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t28_reddit_control_audit_seed_sweep.csv")
    parser.add_argument("--report", default="experiments/summaries/t28_reddit_control_audit_summary.md")
    args = parser.parse_args()
    csv_path = write_control_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
