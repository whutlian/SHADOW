from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.products_recovery import PRODUCTS_FULLGRAPH_TEACHER
from shadow_hgc.sft.products_recovery_t26 import (
    budget_to_json,
    compute_p0_recovery_diagnostics,
    mixed_class_budget,
    per_class_collapse_report,
)
from shadow_hgc.sft.t26_contract import T26_PRODUCTS_DIAGNOSTICS, T26_PRODUCTS_METHODS, T26_REQUIRED_FIELDS, make_t26_row
from shadow_hgc.train.lazy_sft_memmap import load_products_labels_and_splits


PRODUCTS_NODES = 2_449_029

PRODUCT_FIELDS = T26_REQUIRED_FIELDS + [
    "diagnostic_id",
    "source_t25_table",
    "source_t25_method",
    "t26_gate",
]

PER_CLASS_FIELDS = ["method", "ratio", "seed", "class_id", "train_count", "budget", "selected_count", "predicted_count", "collapsed"]


def _float_or_blank(value: Any) -> float | str:
    if value in {"", None}:
        return ""
    return float(value)


def _int_or_blank(value: Any) -> int | str:
    if value in {"", None}:
        return ""
    return int(float(value))


def _read_t25_rows(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    for row in rows:
        row["source_t25_table"] = str(path)
    return rows


def _read_long_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path in {"", None}:
        return []
    target = Path(path)
    if not target.exists():
        return []
    rows = read_csv(target)
    for row in rows:
        row["source_long_table"] = str(target)
    return rows


def _best_t25_product_row(rows: list[dict[str, str]], *, ratio: float, method_contains: str | None = None) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get("dataset") == "ogbn-products" and abs(float(row.get("requested_full_node_ratio", 0.0)) - float(ratio)) < 1e-12]
    if method_contains is not None:
        candidates = [row for row in candidates if method_contains in row.get("method", "")]
    candidates = [row for row in candidates if row.get("accuracy") not in {"", None}]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("accuracy", 0.0)))


def _long_product_row(rows: list[dict[str, str]], *, method: str, ratio: float | None = None) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get("dataset") == "ogbn-products" and row.get("method") == method]
    if ratio is not None:
        candidates = [row for row in candidates if row.get("requested_full_node_ratio") not in {"", None} and abs(float(row["requested_full_node_ratio"]) - float(ratio)) < 1e-12]
    if not candidates:
        return None
    return candidates[-1]


def _row_metric(row: dict[str, str] | None, *fields: str) -> float | str:
    if row is None:
        return ""
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return float(value)
    return ""


def _row_int(row: dict[str, str] | None, *fields: str) -> int | str:
    if row is None:
        return ""
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return int(float(value))
    return ""


def _labels_or_none(products_root: str | Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    try:
        labels, train_rows, _valid_rows, _test_rows = load_products_labels_and_splits(products_root)
        return labels, train_rows
    except Exception:
        return None, None


def build_products_outputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    t25_rows = _read_t25_rows(args.t25_products_csv)
    long_rows = _read_long_rows(getattr(args, "long_results_csv", None))
    labels, train_rows = _labels_or_none(args.products_root)
    num_classes = 47 if labels is None else int(labels.max().item()) + 1
    ratios = [float(value) for value in args.ratios]
    diagnostic_rows: list[dict[str, Any]] = []
    uca_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    source_ref = str(args.t25_products_csv)

    for ratio in ratios:
        total_budget = max(num_classes, int(round(PRODUCTS_NODES * ratio)))
        budget = (
            {cls: max(1, total_budget // num_classes) for cls in range(num_classes)}
            if labels is None or train_rows is None
            else mixed_class_budget(labels, train_rows, total_budget=total_budget, ratio=ratio, num_classes=num_classes, seed=int(args.seed))
        )
        if labels is not None and train_rows is not None:
            selected_for_report = train_rows[:0]
            report = per_class_collapse_report(labels[train_rows], selected_for_report, None, num_classes=num_classes, budget=budget)
        else:
            report = [
                {"class_id": cls, "train_count": "", "budget": budget.get(cls, ""), "selected_count": "", "predicted_count": "", "collapsed": True}
                for cls in range(num_classes)
            ]
        for item in report:
            per_class_rows.append({"method": "products_uca_hybrid_balanced_trainer", "ratio": ratio, "seed": int(args.seed), **item})
        best_existing = _best_t25_product_row(t25_rows, ratio=ratio)
        p0a_source = _long_product_row(long_rows, method="P0a_alltrain_condensed_trainer_parity")
        p0b_source = _long_product_row(long_rows, method="P0b_selected_prototype_self_fit", ratio=ratio)
        p0_diag = compute_p0_recovery_diagnostics(
            alltrain_acc=_row_metric(p0a_source, "p0a_alltrain_acc", "accuracy"),
            self_fit_acc=_row_metric(p0b_source, "p0b_self_fit_acc", "accuracy"),
            normalization_match=bool(best_existing and best_existing.get("products_diag_memmap_row_order_matches_node_id") == "True" and best_existing.get("products_diag_masks_aligned") == "True"),
            predicted_class_count=_row_int(p0a_source, "predicted_class_count", "predicted_classes") or ("" if best_existing is None else best_existing.get("predicted_class_count", "")),
            num_classes=num_classes,
        )
        for diag_id in T26_PRODUCTS_DIAGNOSTICS:
            diag_source = _long_product_row(long_rows, method=diag_id, ratio=ratio) or _long_product_row(long_rows, method=diag_id)
            failure = {
                "P0a_alltrain_condensed_trainer_parity": "P0a_condensed_trainer_parity_not_rerun",
                "P0b_selected_prototype_self_fit": "P0b_self_fit_not_rerun",
                "P0c_same_budget_random_subset": "P0c_random_subset_not_rerun",
                "P0d_nearest_prototype_oracle": "P0d_oracle_not_rerun",
                "P0e_per_class_collapse_report": "per_class_report_schema_written_waiting_for_real_selection_and_predictions",
                "P0f_feature_normalization_parity": "normalization_parity_from_existing_manifest",
            }[diag_id]
            status = "ready_not_run" if diag_id == "P0e_per_class_collapse_report" else ("completed_diagnostic" if diag_id == "P0f_feature_normalization_parity" else "ready_not_run")
            if diag_source is not None:
                status = diag_source.get("status", "completed_long") or "completed_long"
                failure = ""
                if diag_id == "P0a_alltrain_condensed_trainer_parity" and not p0_diag["p0a_passed"]:
                    failure = "P0a_gate_not_met"
                if diag_id == "P0b_selected_prototype_self_fit" and not p0_diag["p0b_passed"]:
                    failure = "P0b_gate_not_met"
            if diag_id == "P0f_feature_normalization_parity" and not p0_diag["p0f_normalization_parity"]:
                status = "blocked"
                failure = "normalization_parity_not_confirmed"
            row = make_t26_row(
                dataset="ogbn-products",
                method=diag_id,
                requested_full_node_ratio=ratio,
                original_total_nodes=PRODUCTS_NODES,
                target_prototypes=total_budget,
                shadow_nodes=0,
                total_condensed_edges=total_budget,
                seed=int(args.seed),
                accuracy=_row_metric(diag_source, "accuracy"),
                macro_f1=_row_metric(diag_source, "macro_f1"),
                predicted_classes=_row_int(diag_source, "predicted_class_count", "predicted_classes"),
                status=status,
                promotion_status="not_promoted",
                promotion_reason="products_recovery_diagnostic_only",
                failure_reason=failure,
                notes="T26 diagnostics do not promote performance rows unless P0a and P0b are rerun and pass.",
                split="sales_ranking",
                per_class_report_path=str(args.per_class_csv),
                class_budget_policy="floor_plus_0.45p_0.35sqrt_0.20uniform",
                class_budget_floor=min(budget.values()) if budget else "",
                class_budget_min=min(budget.values()) if budget else "",
                class_budget_max=max(budget.values()) if budget else "",
                class_budget_json=budget_to_json(budget),
                source_t25_table=source_ref,
                source_t25_method="" if best_existing is None else best_existing.get("method", ""),
                diagnostic_id=diag_id,
                t26_gate=failure,
                p0d_prototype_oracle_acc=_row_metric(diag_source, "p0d_prototype_oracle_acc", "prototype_oracle_acc"),
                p0d_centroid_oracle_acc=_row_metric(diag_source, "p0d_centroid_oracle_acc", "centroid_oracle_acc"),
                peak_cpu_ram=current_cpu_ram_bytes() / (1024**3),
                peak_gpu_ram=current_gpu_ram_bytes() / (1024**3),
                **p0_diag,
            )
            diagnostic_rows.append(row)

        for method in T26_PRODUCTS_METHODS:
            source_row = _best_t25_product_row(t25_rows, ratio=ratio, method_contains="P1") if method.startswith("products_cb") else None
            long_source = _long_product_row(long_rows, method=method, ratio=ratio)
            p0_passed = bool(p0_diag["p0a_passed"]) and bool(p0_diag["p0b_passed"])
            p0_failure = "blocked_by_P0a_P0b_gate"
            status = "blocked_by_P0_gate"
            failure_reason = p0_failure
            promotion_reason = p0_failure
            if p0_passed and long_source is not None:
                status = long_source.get("status", "completed_long") or "completed_long"
                failure_reason = "products_t26_gate_not_met"
                promotion_reason = "products_t26_gate_not_met"
            elif p0_passed:
                status = "ready_not_run"
                failure_reason = "long_experiment_not_run"
                promotion_reason = "long_experiment_not_run"
            row = make_t26_row(
                dataset="ogbn-products",
                method=method,
                requested_full_node_ratio=ratio,
                original_total_nodes=PRODUCTS_NODES,
                target_prototypes=total_budget,
                shadow_nodes=0,
                total_condensed_edges=total_budget,
                seed=int(args.seed),
                accuracy=_row_metric(long_source, "accuracy") if p0_passed else "",
                macro_f1=_row_metric(long_source, "macro_f1") if p0_passed else "",
                predicted_classes=_row_int(long_source, "predicted_class_count", "predicted_classes") if p0_passed else "",
                status=status,
                promotion_status="not_promoted",
                promotion_reason=promotion_reason,
                failure_reason=failure_reason,
                notes="UCA/product recovery rows are not promoted or assigned T26 performance until P0a >= 0.74 and P0b >= 0.95 are actually satisfied.",
                split="sales_ranking",
                per_class_report_path=str(args.per_class_csv),
                class_budget_policy="floor_plus_0.45p_0.35sqrt_0.20uniform",
                class_budget_floor=min(budget.values()) if budget else "",
                class_budget_min=min(budget.values()) if budget else "",
                class_budget_max=max(budget.values()) if budget else "",
                class_budget_json=budget_to_json(budget),
                coverage_gap_l1=_row_metric(long_source, "coverage_gap_l1"),
                coverage_gap_l2=_row_metric(long_source, "coverage_gap_l2"),
                uca_num_domains=_row_int(long_source, "uca_num_domains") or int(args.uca_domains),
                uca_domain_seed=_row_int(long_source, "uca_domain_seed") or int(args.seed),
                uca_uses_valid_test_labels=False,
                trainer_recipe="balanced_adamw_label_smoothing_mixup" if "balanced_trainer" in method or "mixup" in method else "standard_adamw",
                trainer_balanced_batches="balanced_trainer" in method,
                trainer_label_smoothing=0.05 if "balanced_trainer" in method else 0.0,
                trainer_mixup_alpha=0.4 if "mixup" in method else 0.0,
                source_t25_table=source_ref,
                source_t25_method="" if source_row is None else source_row.get("method", ""),
                t26_gate=p0_failure,
                **p0_diag,
            )
            uca_rows.append(row)
    return diagnostic_rows, uca_rows, per_class_rows


def write_products_outputs(args: argparse.Namespace) -> dict[str, Path]:
    diagnostics, uca_rows, per_class_rows = build_products_outputs(args)
    diag_csv = write_csv(args.diagnostics_csv, diagnostics, PRODUCT_FIELDS)
    uca_csv = write_csv(args.uca_csv, uca_rows, PRODUCT_FIELDS)
    per_class_csv = write_csv(args.per_class_csv, per_class_rows, PER_CLASS_FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Products Recovery Notes",
            "",
            "- Product rows are blocked from promotion until P0a all-train condensed-trainer parity and P0b selected-prototype self-fit are run and pass.",
            "- T25 replay rows are used only as source diagnostics; they are not relabeled as T26 promoted results.",
            "- UCA leakage flag remains false for all generated rows.",
            "",
            "## Diagnostics",
            "",
            *markdown_table(diagnostics, ["requested_full_node_ratio", "method", "status", "p0a_passed", "p0b_passed", "p0f_normalization_parity", "failure_reason"]),
            "",
            "## UCA Sweep",
            "",
            *markdown_table(uca_rows, ["requested_full_node_ratio", "method", "status", "accuracy", "macro_f1", "predicted_class_count", "promotion_status", "failure_reason"]),
            "",
            f"- Diagnostics CSV: `{diag_csv}`",
            f"- UCA CSV: `{uca_csv}`",
            f"- Per-class CSV: `{per_class_csv}`",
        ],
    )
    return {"diagnostics": diag_csv, "uca": uca_csv, "per_class": per_class_csv, "report": Path(args.report)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T26 products recovery diagnostics and UCA sweep tables.")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--t25-products-csv", default="experiments/tables/t25_products_recovery_ladder_seed42.csv")
    parser.add_argument("--long-results-csv", default="experiments/tables/t26_products_long_experiments_seed42.csv")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0025, 0.005])
    parser.add_argument("--uca-domains", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics-csv", default="experiments/tables/t26_products_recovery_diagnostics_seed42.csv")
    parser.add_argument("--per-class-csv", default="experiments/tables/t26_products_per_class_report_seed42.csv")
    parser.add_argument("--uca-csv", default="experiments/tables/t26_products_uca_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_products_recovery_notes.md")
    args = parser.parse_args()
    outputs = write_products_outputs(args)
    print(json.dumps({"status": "completed", "outputs": {key: str(value) for key, value in outputs.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
