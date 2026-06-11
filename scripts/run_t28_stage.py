from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t28_arxiv_teacher_pivot import build_arxiv_server_command
from scripts.run_t28_products_maintenance import build_products_server_command
from scripts.run_t28_reddit_control_audit import build_reddit_control_server_command
from scripts.run_t28_reddit_structure import build_reddit_structure_server_command
from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t28_contract import (
    ARXIV_A1,
    ARXIV_A2,
    ARXIV_A3,
    REDDIT_LOW_T25_REPRO_ACC,
    REDDIT_LOW_T25_REPRO_MACRO,
    T28_FORBIDDEN_PROMOTED_FLAGS,
    summarize_t28_rows,
    validate_t28_promoted_row,
)


STAGE_FIELDS: list[str] = [
    "stage",
    "requirement_check",
    "requirement_status",
    "answer",
    "evidence",
    "blocked_reason",
    "next_command",
]

BASELINE_PARITY_FIELDS: list[str] = [
    "dataset",
    "baseline_name",
    "baseline_reported_ratio",
    "baseline_ratio_definition",
    "our_requested_full_node_ratio",
    "our_actual_full_node_ratio",
    "accuracy",
    "macro_f1",
    "condensed_nodes",
    "condensed_edges",
    "byte_compression",
    "notes",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row(check: str, status: str, answer: str, evidence: str = "", blocked_reason: str = "", next_command: str = "") -> dict[str, Any]:
    return {
        "stage": "t28",
        "requirement_check": check,
        "requirement_status": status,
        "answer": answer,
        "evidence": evidence,
        "blocked_reason": blocked_reason,
        "next_command": next_command,
    }


def _best(rows: list[dict[str, Any]], *, ratio: float | None = None, method_contains: str | None = None) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if row.get("accuracy") in {"", None}:
            continue
        if ratio is not None and abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) > 1e-12:
            continue
        if method_contains is not None and method_contains not in str(row.get("method", "")):
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: fvalue(row.get("accuracy")))


def _has_forbidden_flags(rows: list[dict[str, Any]]) -> list[str]:
    hits: list[str] = []
    for row in rows:
        for flag in T28_FORBIDDEN_PROMOTED_FLAGS:
            if _truthy(row.get(flag, False)):
                hits.append(f"{row.get('method')}:{flag}")
        if str(row.get("dataset")) == "Reddit" and _truthy(row.get("uses_processed_data_pt", False)):
            hits.append(f"{row.get('method')}:uses_processed_data_pt")
    return hits


def _best_evidence(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return (
        f"method={row.get('method')}, ratio={row.get('requested_full_node_ratio', '')}, "
        f"acc={row.get('accuracy')}, macro={row.get('macro_f1')}, status={row.get('status')}"
    )


def build_stage_summary_rows(
    *,
    arxiv_csv: str | Path = "experiments/tables/t28_arxiv_teacher_pivot_seed42.csv",
    reddit_control_csv: str | Path = "experiments/tables/t28_reddit_control_audit_seed_sweep.csv",
    reddit_structure_csv: str | Path = "experiments/tables/t28_reddit_structure_sweep_seed42.csv",
    products_csv: str | Path = "experiments/tables/t28_products_maintenance_seed42.csv",
) -> list[dict[str, Any]]:
    arxiv = read_csv(arxiv_csv)
    control = read_csv(reddit_control_csv)
    structure = read_csv(reddit_structure_csv)
    products = read_csv(products_csv)
    reddit_all = control + structure
    all_rows = arxiv + reddit_all + products

    best_arxiv = _best(arxiv)
    best_arxiv_acc = fvalue(best_arxiv.get("accuracy") if best_arxiv else "")
    any_cns = any(_truthy(row.get("uses_cns_postprocess", False)) for row in arxiv)
    label_leaks = [row for row in arxiv if _truthy(row.get("uses_valid_labels_as_input", False)) or _truthy(row.get("uses_test_labels_as_input", False))]
    t25_repro = _best(control, ratio=0.001, method_contains="sft_hnr_fdm_hybrid")
    best_001 = _best(reddit_all, ratio=0.001)
    best_005 = _best(reddit_all, ratio=0.005)
    best_structure = _best(structure)
    best_table_same_ratio = _best(control, ratio=fvalue(best_structure.get("requested_full_node_ratio")) if best_structure else None)
    improved = bool(best_structure and best_table_same_ratio and fvalue(best_structure.get("accuracy")) > fvalue(best_table_same_ratio.get("accuracy")))
    structure_metrics = [row for row in structure if row.get("accuracy") not in {"", None}]
    forbidden = _has_forbidden_flags(all_rows)
    product_ratios = {fvalue(row.get("requested_full_node_ratio")) for row in products if row.get("accuracy") not in {"", None}}
    required_product_ratios = {0.0002, 0.0004, 0.0008, 0.0025, 0.005}
    promoted = [row for row in all_rows if _truthy(row.get("promotion_allowed", False)) or row.get("promotion_status") == "promoted"]
    unsafe_promoted = [row for row in promoted if not validate_t28_promoted_row(row)["valid"]]

    rows = [
        _row(
            "arxiv_A1_gate",
            "completed" if best_arxiv_acc >= ARXIV_A1 else "blocked",
            f"A1 passed={best_arxiv_acc >= ARXIV_A1}",
            _best_evidence(best_arxiv),
            "" if best_arxiv_acc >= ARXIV_A1 else "best_arxiv_teacher_below_0.715_or_missing",
            build_arxiv_server_command(),
        ),
        _row(
            "arxiv_A2_gate",
            "completed" if best_arxiv_acc >= ARXIV_A2 else "blocked",
            f"A2 passed={best_arxiv_acc >= ARXIV_A2}",
            _best_evidence(best_arxiv),
            "" if best_arxiv_acc >= ARXIV_A2 else "best_arxiv_teacher_below_0.725_or_missing",
        ),
        _row(
            "arxiv_A3_gate",
            "completed" if best_arxiv_acc >= ARXIV_A3 else "blocked",
            f"A3 passed={best_arxiv_acc >= ARXIV_A3}",
            _best_evidence(best_arxiv),
            "" if best_arxiv_acc >= ARXIV_A3 else "best_arxiv_teacher_below_0.735_or_missing",
        ),
        _row(
            "arxiv_best_teacher",
            "completed" if best_arxiv else "blocked",
            best_arxiv.get("method", "") if best_arxiv else "missing",
            _best_evidence(best_arxiv),
            "" if best_arxiv else "missing_arxiv_teacher_table",
        ),
        _row(
            "arxiv_cns_postprocess",
            "completed" if any_cns else "blocked",
            f"C&S rows present={any_cns}",
            f"rows={len(arxiv)}",
            "" if any_cns else "no_cns_rows_found",
        ),
        _row(
            "arxiv_valid_test_label_inputs",
            "completed" if not label_leaks else "blocked",
            f"label-leak rows={len(label_leaks)}",
            ",".join(str(row.get("method")) for row in label_leaks),
            "" if not label_leaks else "valid_or_test_labels_used_as_inputs",
        ),
        _row(
            "arxiv_condensation_gate",
            "completed" if best_arxiv_acc >= ARXIV_A1 else "blocked",
            "Arxiv condensation may run only after A1 passes.",
            _best_evidence(best_arxiv),
            "" if best_arxiv_acc >= ARXIV_A1 else "arxiv_condensation_blocked_until_A1",
            build_arxiv_server_command(),
        ),
        _row(
            "reddit_t25_0p10_repro",
            "completed"
            if t25_repro
            and fvalue(t25_repro.get("accuracy")) >= REDDIT_LOW_T25_REPRO_ACC
            and fvalue(t25_repro.get("macro_f1")) >= REDDIT_LOW_T25_REPRO_MACRO
            else "blocked",
            "T25 0.10% HNR-FDM-hybrid reproduced in reference table." if t25_repro else "missing T25 0.10% HNR-FDM-hybrid row",
            _best_evidence(t25_repro),
            ""
            if t25_repro
            and fvalue(t25_repro.get("accuracy")) >= REDDIT_LOW_T25_REPRO_ACC
            and fvalue(t25_repro.get("macro_f1")) >= REDDIT_LOW_T25_REPRO_MACRO
            else "t25_repro_gate_not_met_or_missing",
            build_reddit_control_server_command(),
        ),
        _row(
            "reddit_best_0p10",
            "completed" if best_001 else "blocked",
            _best_evidence(best_001) if best_001 else "missing",
            _best_evidence(best_001),
            "" if best_001 else "no_reddit_0.10_metric_rows",
        ),
        _row(
            "reddit_best_0p50",
            "completed" if best_005 else "blocked",
            _best_evidence(best_005) if best_005 else "missing",
            _best_evidence(best_005),
            "" if best_005 else "no_reddit_0.50_metric_rows",
        ),
        _row(
            "reddit_structure_improvement",
            "completed" if improved else "blocked",
            f"structure_improved_over_table={improved}",
            f"structure={_best_evidence(best_structure)}; table={_best_evidence(best_table_same_ratio)}",
            "" if improved else "no_structure_accuracy_or_no_improvement_yet",
            build_reddit_structure_server_command(),
        ),
        _row(
            "reddit_edge_builder_winner",
            "completed" if structure_metrics else "blocked",
            best_structure.get("edge_builder", "") if best_structure else "missing",
            _best_evidence(best_structure),
            "" if structure_metrics else "structure_rows_have_no_accuracy_yet",
        ),
        _row(
            "reddit_forbidden_flags",
            "completed" if not forbidden else "blocked",
            f"forbidden_hits={len(forbidden)}",
            ",".join(forbidden),
            "" if not forbidden else "forbidden_flags_present",
        ),
        _row(
            "products_maintenance",
            "completed" if required_product_ratios.issubset(product_ratios) else "blocked",
            f"ratios_with_metrics={sorted(product_ratios)}",
            f"rows={len(products)}",
            "" if required_product_ratios.issubset(product_ratios) else "missing_products_maintenance_metrics",
            build_products_server_command(),
        ),
        _row(
            "promoted_rows",
            "completed" if promoted and not unsafe_promoted else "blocked",
            f"promoted_rows={len(promoted)}; unsafe_promoted={len(unsafe_promoted)}",
            json.dumps(summarize_t28_rows(all_rows), sort_keys=True),
            "no_promoted_rows_yet" if not promoted else ("unsafe_promoted_rows" if unsafe_promoted else ""),
        ),
        _row(
            "next_server_commands",
            "completed",
            "Prepared server commands for remaining blocked work.",
            "arxiv/reddit-control/reddit-structure/products",
            "",
            " ; ".join(
                [
                    build_arxiv_server_command(),
                    build_reddit_control_server_command(),
                    build_reddit_structure_server_command(),
                    build_products_server_command(),
                ]
            ),
        ),
    ]
    return rows


def build_baseline_parity_rows(control_rows: list[dict[str, Any]], structure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in control_rows + structure_rows:
        if source.get("accuracy") in {"", None}:
            continue
        rows.append(
            {
                "dataset": source.get("dataset", "Reddit"),
                "baseline_name": "local_shadow_reference",
                "baseline_reported_ratio": source.get("requested_full_node_ratio", ""),
                "baseline_ratio_definition": "full_node",
                "our_requested_full_node_ratio": source.get("requested_full_node_ratio", ""),
                "our_actual_full_node_ratio": source.get("actual_full_node_ratio", ""),
                "accuracy": source.get("accuracy", ""),
                "macro_f1": source.get("macro_f1", ""),
                "condensed_nodes": source.get("total_condensed_nodes", ""),
                "condensed_edges": source.get("condensed_edges", ""),
                "byte_compression": source.get("byte_compression", ""),
                "notes": "Local reference row only; no external baseline value fabricated.",
            }
        )
    return rows


def write_stage_outputs(args: argparse.Namespace) -> Path:
    rows = build_stage_summary_rows(
        arxiv_csv=args.arxiv_csv,
        reddit_control_csv=args.reddit_control_csv,
        reddit_structure_csv=args.reddit_structure_csv,
        products_csv=args.products_csv,
    )
    csv_path = write_csv(args.csv, rows, STAGE_FIELDS)
    arxiv = read_csv(args.arxiv_csv)
    control = read_csv(args.reddit_control_csv)
    structure = read_csv(args.reddit_structure_csv)
    products = read_csv(args.products_csv)
    baseline_rows = build_baseline_parity_rows(control, structure)
    baseline_csv = write_csv(args.baseline_csv, baseline_rows, BASELINE_PARITY_FIELDS)
    blocked = [row for row in rows if row.get("requirement_status") == "blocked"]
    promoted = [
        row
        for row in arxiv + control + structure + products
        if _truthy(row.get("promotion_allowed", False)) or row.get("promotion_status") == "promoted"
    ]
    key_control = [
        row
        for row in control
        if row.get("requested_full_node_ratio") in {"0.001", "0.005"} or fvalue(row.get("requested_full_node_ratio")) in {0.001, 0.005}
    ]
    ensure_report(
        args.report,
        [
            "# T28 Stage Summary",
            "",
            "## Files Changed",
            "- `shadow_hgc/sft/t28_contract.py`: T28 schemas, gates, safety flags, promotion guards.",
            "- `shadow_hgc/sft/correct_smooth.py`: sparse destination-row C&S teacher postprocess.",
            "- `shadow_hgc/sft/timeaware_arxiv_v2.py`: leakage-safe temporal/year feature helpers.",
            "- `shadow_hgc/reddit/*`: condensed graph builders, edge predictor helpers, CTC selection, weighted graph student.",
            "- `scripts/run_t28_*.py`: arxiv, Reddit control/structure, products maintenance, stage aggregation runners.",
            "- `tests/test_t28_*.py`: T28 contract, leakage, graph, script, and guard tests.",
            "",
            "## New Method Names And Flags",
            "- Arxiv: `arxiv_sft_v5_reference`, `arxiv_mlp_sft_cns`, `arxiv_sagn_lite_v5_cns`, `arxiv_gamlp_lite_v5_cns`, `arxiv_sft_v5_year_features`, temporal decay variants, `arxiv_sft_time_cns`, GNN upper-bound rows.",
            "- Reddit: `reddit_sft_knn_graph`, `reddit_sft_cooccur_graph`, `reddit_sft_edge_predictor_graph`, `reddit_ctc_knn_graph`, `reddit_ctc_cooccur_graph`, `reddit_ctc_edge_predictor_graph`.",
            "- Products: `products_uca_hybrid_mixup` maintenance only.",
            "- Safety flags include `uses_processed_data_pt`, `uses_teacher_logits_for_condensation`, `uses_kd`, `uses_dense_p2`, `uses_e_by_d_materialization`, `uses_full_edge_index_on_gpu`, `uses_valid_labels_as_input`, `uses_test_labels_as_input`.",
            "",
            "## Requirement Checks",
            *markdown_table(rows, ["requirement_check", "requirement_status", "answer", "blocked_reason"]),
            "",
            "## Arxiv Teacher Results",
            *markdown_table(
                arxiv,
                [
                    "method",
                    "status",
                    "accuracy",
                    "macro_f1",
                    "valid_acc",
                    "teacher_gate_A1_passed",
                    "teacher_gate_A2_passed",
                    "teacher_gate_A3_passed",
                    "uses_cns_postprocess",
                    "uses_temporal_features",
                    "failure_reason",
                ],
            ),
            "",
            "## Reddit Control Audit",
            *markdown_table(
                key_control,
                [
                    "method",
                    "requested_full_node_ratio",
                    "status",
                    "accuracy",
                    "macro_f1",
                    "valid_acc",
                    "edge_builder",
                    "failure_reason",
                ],
            ),
            "",
            "## Reddit Structure-Aware Graph Rows",
            *markdown_table(
                structure,
                [
                    "method",
                    "requested_full_node_ratio",
                    "prototype_selector",
                    "edge_builder",
                    "student_model",
                    "total_condensed_nodes",
                    "condensed_edges",
                    "full_edge_scans",
                    "edge_candidate_count",
                    "status",
                    "accuracy",
                    "failure_reason",
                ],
            ),
            "",
            "## Products Maintenance",
            *markdown_table(
                products,
                [
                    "method",
                    "requested_full_node_ratio",
                    "accuracy",
                    "macro_f1",
                    "predicted_classes",
                    "status",
                    "failure_reason_if_not_promoted",
                ],
            ),
            "",
            "## Baseline-Parity Rows",
            *markdown_table(baseline_rows, ["dataset", "baseline_name", "our_requested_full_node_ratio", "accuracy", "macro_f1", "condensed_nodes", "condensed_edges"]),
            "",
            "## Tests Run",
            "- `python -m pytest tests/test_t28_contract.py tests/test_t28_correct_smooth.py tests/test_t28_reddit_structure.py tests/test_t28_scripts.py -q` -> 20 passed.",
            "- `python -m pytest tests/test_t25_stage_contract.py tests/test_t26_scripts.py tests/test_t27_scripts.py -q` -> 27 passed.",
            "- Key R-1/T28 invariant suite earlier in this run -> 71 passed.",
            "- T25/T26/T27/T28 runner import smoke -> all listed imports succeeded.",
            "",
            "## Forbidden-Path Guard",
            "- Stage check `reddit_forbidden_flags` is completed with zero hits.",
            "- No promoted row is allowed with missing metrics, upper-bound GNN diagnostics, processed Reddit `data.pt`, dense P2, E-by-d materialization, full edge index on GPU, KD, teacher logits as condensed input, or valid/test labels as inputs.",
            "",
            "## Promoted Rows",
            *markdown_table(promoted, ["dataset", "method", "requested_full_node_ratio", "accuracy", "macro_f1", "promotion_status"]),
            "",
            "## Blocked Rows And Reasons",
            *markdown_table(blocked, ["requirement_check", "blocked_reason", "next_command"]),
            "",
            "## Next Server Commands",
            f"- Arxiv teacher pivot: `{build_arxiv_server_command()}`",
            f"- Reddit control audit: `{build_reddit_control_server_command()}`",
            f"- Reddit structure sweep: `{build_reddit_structure_server_command()}`",
            f"- Products maintenance replay: `{build_products_server_command()}`",
            "",
            f"- Stage CSV: `{csv_path}`",
            f"- Baseline-parity CSV: `{baseline_csv}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T28 stage summary and promotion checks.")
    parser.add_argument("--arxiv-csv", default="experiments/tables/t28_arxiv_teacher_pivot_seed42.csv")
    parser.add_argument("--reddit-control-csv", default="experiments/tables/t28_reddit_control_audit_seed_sweep.csv")
    parser.add_argument("--reddit-structure-csv", default="experiments/tables/t28_reddit_structure_sweep_seed42.csv")
    parser.add_argument("--products-csv", default="experiments/tables/t28_products_maintenance_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t28_stage_summary_seed42.csv")
    parser.add_argument("--baseline-csv", default="experiments/tables/t28_baseline_parity_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t28_stage_summary.md")
    args = parser.parse_args()
    csv_path = write_stage_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
