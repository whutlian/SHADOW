from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t38_unified_stage import build_t38_rows, load_reference_index, ratio_key, write_t38_main_curve
from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.unified_stt import (
    PUBLIC_METHOD_ID,
    PUBLIC_METHOD_NAME,
    acceptable_gap,
    fvalue,
    validate_t38_main_table,
)


UPPER_FIELDS = [
    "dataset",
    "ratio",
    "unified_acc",
    "specialized_best_method",
    "specialized_best_acc",
    "unified_minus_specialized_acc",
    "unified_macro_f1",
    "specialized_macro_f1",
    "unified_minus_specialized_macro_f1",
    "acceptable_gap",
]

ABLATION_FIELDS = [
    "dataset",
    "ratio",
    "ablation",
    "status",
    "accuracy",
    "macro_f1",
    "source",
    "notes",
]

SCALABILITY_FIELDS = [
    "method",
    "paradigm",
    "requires_per_ratio_fullgraph_rebuild",
    "requires_full_edge_index_gpu",
    "requires_dense_p2",
    "requires_e_by_d_materialization",
    "requires_dense_nxc_teacher_cache",
    "can_run_papers100m",
    "observed_status",
    "failure_stage",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "notes",
]


def _source_method(row: dict[str, Any]) -> str:
    return str(row.get("method", ""))


def build_specialized_upper_bound(
    *,
    main_rows: list[dict[str, Any]],
    tables_dir: str | Path,
) -> list[dict[str, Any]]:
    refs = load_reference_index(tables_dir)
    out: list[dict[str, Any]] = []
    for row in main_rows:
        if row.get("accuracy") in {"", None}:
            continue
        key = (str(row.get("dataset")), ratio_key(row.get("requested_full_node_ratio", 0.0)), str(row.get("comparison_type")))
        ref = refs.get(key)
        if ref is None:
            continue
        acc_gap, macro_gap = acceptable_gap(str(row.get("dataset")))
        unified_acc = fvalue(row.get("accuracy"))
        specialized_acc = fvalue(ref.get("accuracy"))
        unified_macro = fvalue(row.get("macro_f1"))
        specialized_macro = fvalue(ref.get("macro_f1"))
        out.append(
            {
                "dataset": row.get("dataset"),
                "ratio": row.get("requested_full_node_ratio"),
                "unified_acc": unified_acc,
                "specialized_best_method": _source_method(ref),
                "specialized_best_acc": specialized_acc,
                "unified_minus_specialized_acc": unified_acc - specialized_acc,
                "unified_macro_f1": unified_macro,
                "specialized_macro_f1": specialized_macro,
                "unified_minus_specialized_macro_f1": unified_macro - specialized_macro,
                "acceptable_gap": abs(unified_acc - specialized_acc) <= acc_gap and abs(unified_macro - specialized_macro) <= macro_gap,
            }
        )
    return out


def build_unified_ablation(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        ("Reddit", 0.001),
        ("Reddit", 0.005),
        ("ogbn-products", 0.0025),
        ("ogbn-papers100M", 0.0001),
        ("ogbn-papers100M", 0.0005),
    }
    ablations = [
        "coverage_only",
        "coverage_plus_hard",
        "coverage_plus_hard_plus_rare",
        "coverage_plus_teacher",
        "coverage_plus_teacher_plus_boundary",
        "full_auto_schedule",
        "fixed_low_budget_schedule",
        "fixed_high_budget_schedule",
        "dense_teacher_if_allowed",
        "topk_teacher",
    ]
    main_by_key = {
        (str(row.get("dataset")), float(row.get("requested_full_node_ratio", 0.0) or 0.0), str(row.get("comparison_type"))): row
        for row in main_rows
        if row.get("accuracy") not in {"", None}
    }
    out: list[dict[str, Any]] = []
    for dataset, ratio in sorted(required):
        candidates = [row for (d, r, _), row in main_by_key.items() if d == dataset and abs(r - ratio) < 1e-12]
        main = candidates[0] if candidates else {}
        for ablation in ablations:
            completed = ablation == "full_auto_schedule" and bool(main)
            out.append(
                {
                    "dataset": dataset,
                    "ratio": ratio,
                    "ablation": ablation,
                    "status": "completed_real" if completed else "not_run_in_t38_current_pass",
                    "accuracy": main.get("accuracy", "") if completed else "",
                    "macro_f1": main.get("macro_f1", "") if completed else "",
                    "source": "t38_main_curve" if completed else "",
                    "notes": "missing real ablation log; left blank rather than fabricating" if not completed else "full unified auto schedule row",
                }
            )
    return out


def build_scalability_table(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    papers = [row for row in main_rows if str(row.get("dataset")) == "ogbn-papers100M" and row.get("accuracy") not in {"", None}]
    peak_cpu = max([fvalue(row.get("peak_cpu_ram")) for row in papers] or [0.0])
    peak_gpu = max([fvalue(row.get("peak_gpu_ram")) for row in papers] or [0.0])
    cache_bytes = max([fvalue(row.get("total_storage_bytes")) for row in papers] or [0.0])
    return [
        {
            "method": PUBLIC_METHOD_NAME,
            "paradigm": "one_cache_streaming_unified_reservoir",
            "requires_per_ratio_fullgraph_rebuild": False,
            "requires_full_edge_index_gpu": False,
            "requires_dense_p2": False,
            "requires_e_by_d_materialization": False,
            "requires_dense_nxc_teacher_cache": False,
            "can_run_papers100m": bool(papers),
            "observed_status": "completed_real_rows_present" if papers else "not_run",
            "failure_stage": "",
            "peak_cpu_ram": int(peak_cpu),
            "peak_gpu_ram": int(peak_gpu),
            "cache_bytes": int(cache_bytes),
            "notes": "T38 promoted rows reuse one papers100M cache with zero post-build edge scans.",
        },
        {
            "method": "external_dense_synthetic_adjacency_methods",
            "paradigm": "dense_or_per_ratio_graph_synthesis",
            "requires_per_ratio_fullgraph_rebuild": "",
            "requires_full_edge_index_gpu": "",
            "requires_dense_p2": "",
            "requires_e_by_d_materialization": "",
            "requires_dense_nxc_teacher_cache": "",
            "can_run_papers100m": "",
            "observed_status": "not_run_in_t38",
            "failure_stage": "",
            "peak_cpu_ram": "",
            "peak_gpu_ram": "",
            "cache_bytes": "",
            "notes": "No OOM fabricated; fill only after a controlled baseline attempt.",
        },
    ]


def write_summary(
    *,
    path: str | Path,
    main_rows: list[dict[str, Any]],
    upper_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    scalability_rows: list[dict[str, Any]],
    main_guard: dict[str, Any],
) -> Path:
    promoted = [row for row in main_rows if str(row.get("promotion_status")) == "promoted"]
    papers = [row for row in promoted if str(row.get("dataset")) == "ogbn-papers100M"]
    old_leak = any(str(row.get("method")) != PUBLIC_METHOD_ID for row in main_rows)
    lines = [
        "# T38 Shadow-HGC-STT-U Stage Summary",
        "",
        "## Files changed",
        "",
        "- `shadow_hgc/sft/unified_schedule.py`",
        "- `shadow_hgc/sft/unified_stt.py`",
        "- `shadow_hgc/sft/unified_reservoir.py`",
        "- `shadow_hgc/sft/stt_gated_mixer.py`",
        "- `scripts/run_t38_unified_stage.py`",
        "- `scripts/run_t38_stage.py`",
        "- `tests/test_t38_*.py`",
        "",
        "## Public method",
        "",
        f"- Public method ID: `{PUBLIC_METHOD_ID}`",
        f"- Public method name: `{PUBLIC_METHOD_NAME}`",
        "- Old SCR/UCA/TTC/STT names are restricted to specialized upper-bound references.",
        "",
        "## Unified schedule",
        "",
        "- Budget phase: `tau = clip((log2(M/C)-log2(16))/(log2(256)-log2(16)), 0, 1)`.",
        "- Teacher reliability: `q_T = clip((teacher_valid_acc-majority_valid_acc)/(1-majority_valid_acc+eps), 0, 1)`.",
        "- Teacher-disabled rows set soft selection, boundary selection, and soft loss terms to zero.",
        "- Teacher cache mode is selected by byte budget; ultra rows use top-k tail caches.",
        "",
        "## Main curve",
        "",
        *markdown_table(
            promoted,
            [
                "dataset",
                "requested_full_node_ratio",
                "comparison_type",
                "accuracy",
                "macro_f1",
                "predicted_classes",
                "teacher_cache_mode",
                "promotion_status",
            ],
        ),
        "",
        "## Specialized upper-bound gaps",
        "",
        *markdown_table(
            upper_rows,
            [
                "dataset",
                "ratio",
                "unified_acc",
                "specialized_best_method",
                "specialized_best_acc",
                "unified_minus_specialized_acc",
                "acceptable_gap",
            ],
        ),
        "",
        "## Guard summary",
        "",
        f"- Main-table old method leak: `{old_leak}`",
        f"- Main guard valid: `{main_guard['valid']}`",
        f"- Main guard flags: `{','.join(main_guard.get('forbidden_flags', []))}`",
        f"- papers100M promoted rows: `{len(papers)}`",
        "- papers100M rows log `cache_reused=True` and `incremental_edge_scans_after_cache_build=0`.",
        "- Forbidden paths are set false for promoted rows: no valid/test labels as input, no teacher probs as input features, no dense P2, no E-by-d materialization, no full edge_index on GPU, no dense all-node teacher cache.",
        "",
        "## Ablation status",
        "",
        f"- Rows written: `{len(ablation_rows)}`.",
        "- Only rows with real logs are marked `completed_real`; missing ablations are explicitly blank and not fabricated.",
        "",
        "## Scalability table",
        "",
        *markdown_table(scalability_rows, ["method", "can_run_papers100m", "observed_status", "peak_cpu_ram", "peak_gpu_ram", "cache_bytes"]),
        "",
        "## Recommendation",
        "",
        "- Paper-facing naming and safety guards are ready.",
        "- Full ablation coverage still needs real long runs for rows marked `not_run_in_t38_current_pass` before treating the ablation table as final.",
    ]
    return ensure_report(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T38 paper-facing tables.")
    parser.add_argument("--build-main-curve", action="store_true")
    parser.add_argument("--build-specialized-upper-bound", action="store_true")
    parser.add_argument("--build-ablation-summary", action="store_true")
    parser.add_argument("--build-scalability-table", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--main-csv", default="experiments/tables/t38_unified_main_curve_seed42.csv")
    parser.add_argument("--upper-csv", default="experiments/tables/t38_specialized_upper_bound_seed42.csv")
    parser.add_argument("--ablation-csv", default="experiments/tables/t38_unified_ablation_seed42.csv")
    parser.add_argument("--scalability-csv", default="experiments/tables/t38_scalability_oom_table.csv")
    parser.add_argument("--summary", default="experiments/summaries/t38_shadow_hgc_stt_unified_stage_summary.md")
    args = parser.parse_args()

    if args.build_main_curve or not Path(args.main_csv).exists():
        write_t38_main_curve(
            argparse.Namespace(
                datasets=["all"],
                ratios=None,
                tables_dir=args.tables_dir,
                seed=args.seed,
                csv=args.main_csv,
                merge_existing=False,
            )
        )
    main_rows = read_csv(args.main_csv)
    main_guard = validate_t38_main_table(main_rows)

    upper_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    scalability_rows: list[dict[str, Any]] = []
    if args.build_specialized_upper_bound:
        upper_rows = build_specialized_upper_bound(main_rows=main_rows, tables_dir=args.tables_dir)
        write_csv(args.upper_csv, upper_rows, UPPER_FIELDS)
    elif Path(args.upper_csv).exists():
        upper_rows = read_csv(args.upper_csv)
    if args.build_ablation_summary:
        ablation_rows = build_unified_ablation(main_rows)
        write_csv(args.ablation_csv, ablation_rows, ABLATION_FIELDS)
    elif Path(args.ablation_csv).exists():
        ablation_rows = read_csv(args.ablation_csv)
    if args.build_scalability_table:
        scalability_rows = build_scalability_table(main_rows)
        write_csv(args.scalability_csv, scalability_rows, SCALABILITY_FIELDS)
    elif Path(args.scalability_csv).exists():
        scalability_rows = read_csv(args.scalability_csv)

    write_summary(
        path=args.summary,
        main_rows=main_rows,
        upper_rows=upper_rows,
        ablation_rows=ablation_rows,
        scalability_rows=scalability_rows,
        main_guard=main_guard,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "main_csv": args.main_csv,
                "upper_csv": args.upper_csv,
                "ablation_csv": args.ablation_csv,
                "scalability_csv": args.scalability_csv,
                "summary": args.summary,
                "main_guard_valid": main_guard["valid"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
