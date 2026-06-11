from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t33_contract import fvalue, summarize_guard


FIELDS = ["requirement_check", "status", "value", "threshold", "evidence", "next_action"]


def _check(name: str, ok: bool, value: Any, threshold: Any = "", evidence: str = "", next_action: str = "") -> dict[str, Any]:
    return {"requirement_check": name, "status": "passed" if ok else "blocked", "value": value, "threshold": threshold, "evidence": evidence, "next_action": next_action}


def _best(rows: list[dict[str, Any]], ratio: float, field: str = "accuracy") -> float:
    return max([fvalue(row.get(field)) for row in rows if abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) < 1e-12] or [0.0])


def build_stage_rows(
    *,
    ratio_curve: list[dict[str, Any]],
    multiseed: list[dict[str, Any]],
    targeted: list[dict[str, Any]],
    cache_ablation: list[dict[str, Any]],
    arxiv: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    products: list[dict[str, Any]],
    ultra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_rows = ratio_curve + targeted + cache_ablation + arxiv + semantic + products + ultra
    guard = summarize_guard(all_rows)
    best_001 = max(_best(ratio_curve, 0.001), _best(targeted, 0.001), _best(cache_ablation, 0.001))
    best_005 = max(_best(ratio_curve, 0.005), _best(targeted, 0.005), _best(cache_ablation, 0.005))
    best_arxiv = max([fvalue(row.get("cns_accuracy", row.get("accuracy"))) for row in arxiv] or [0.0])
    topk_rows = [row for row in cache_ablation if str(row.get("teacher_cache_mode", "")).startswith("topk")]
    dense_001 = _best([row for row in cache_ablation if row.get("teacher_cache_mode") == "dense_fp16"], 0.001)
    topk8_001 = _best([row for row in cache_ablation if "topk8" in str(row.get("teacher_cache_mode", ""))], 0.001)
    ultra_promoted = sum(1 for row in ultra if str(row.get("promotion_status", "")) == "promoted")
    return [
        _check("reddit_ratio_curve_rows_present", len(ratio_curve) > 0, len(ratio_curve), ">0"),
        _check("reddit_0p10_first_gate", best_001 >= 0.923, best_001, ">=0.923"),
        _check("reddit_0p10_main_gate", best_001 >= 0.930, best_001, ">=0.930"),
        _check("reddit_0p50_improves_t32", best_005 > 0.9372744735472057, best_005, ">0.9372744735472057"),
        _check("reddit_0p50_first_gate", best_005 >= 0.938, best_005, ">=0.938"),
        _check("teacher_cache_ablation_logged", len(cache_ablation) > 0 and len(topk_rows) > 0, len(cache_ablation), "dense+topk rows"),
        _check("topk8_drop_within_0p2pp_when_available", dense_001 == 0.0 or topk8_001 == 0.0 or dense_001 - topk8_001 <= 0.002, json.dumps({"dense_001": dense_001, "topk8_001": topk8_001}), "<=0.002"),
        _check("arxiv_raw_mlp_cns_0p700", best_arxiv >= 0.700, best_arxiv, ">=0.700"),
        _check("arxiv_teacher_gate_0p715", best_arxiv >= 0.715, best_arxiv, ">=0.715"),
        _check("arxiv_semantic_cache_available", any(str(row.get("semantic_cache_path", "")) and row.get("status") != "blocked" for row in semantic), len(semantic), "validated cache"),
        _check("products_maintenance_rows_present", len(products) > 0, len(products), ">0"),
        _check("ultra_topk_planner_promoted", ultra_promoted > 0, ultra_promoted, ">0"),
        _check("forbidden_guard_hits", int(guard["unsafe_promoted_rows"]) == 0, guard["unsafe_promoted_rows"], "0", evidence=json.dumps(guard, sort_keys=True)),
    ]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_stage_rows(
        ratio_curve=read_csv(args.ratio_curve_csv),
        multiseed=read_csv(args.multiseed_csv),
        targeted=read_csv(args.targeted_csv),
        cache_ablation=read_csv(args.cache_ablation_csv),
        arxiv=read_csv(args.arxiv_csv),
        semantic=read_csv(args.semantic_csv),
        products=read_csv(args.products_csv),
        ultra=read_csv(args.ultra_csv),
    )
    csv_path = write_csv(args.csv, rows, FIELDS)
    ensure_report(args.report, ["# T33 Stage Summary", "", *markdown_table(rows, ["requirement_check", "status", "value", "threshold"]), "", f"- CSV: `{csv_path}`"])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T33 stage checks.")
    parser.add_argument("--ratio-curve-csv", default="experiments/tables/t33_reddit_ttcpp_ratio_curve.csv")
    parser.add_argument("--multiseed-csv", default="experiments/tables/t33_reddit_ttcpp_multiseed.csv")
    parser.add_argument("--targeted-csv", default="experiments/tables/t33_reddit_ttcpp_targeted_0p50.csv")
    parser.add_argument("--cache-ablation-csv", default="experiments/tables/t33_reddit_teacher_cache_ablation.csv")
    parser.add_argument("--arxiv-csv", default="experiments/tables/t33_arxiv_cns_forensic.csv")
    parser.add_argument("--semantic-csv", default="experiments/tables/t33_arxiv_semantic_sft.csv")
    parser.add_argument("--products-csv", default="experiments/tables/t33_products_maintenance_multiseed.csv")
    parser.add_argument("--ultra-csv", default="experiments/tables/t33_ultra_ttcpp_planner.csv")
    parser.add_argument("--csv", default="experiments/tables/t33_stage_summary.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_stage_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
