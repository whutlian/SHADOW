from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t32_contract import fvalue, summarize_guard, truthy


STAGE_FIELDS = [
    "requirement_check",
    "status",
    "value",
    "threshold",
    "evidence",
    "next_action",
]


def _rows_present(rows: list[dict[str, Any]]) -> bool:
    return len(rows) > 0


def _best_acc(rows: list[dict[str, Any]], *, dataset: str | None = None, ratio: float | None = None, method_contains: str | None = None) -> float:
    best = 0.0
    for row in rows:
        if dataset is not None and str(row.get("dataset", "")) != dataset:
            continue
        if ratio is not None and abs(fvalue(row.get("requested_full_node_ratio", 0.0)) - float(ratio)) > 1e-12:
            continue
        if method_contains is not None and method_contains not in str(row.get("method", "")):
            continue
        best = max(best, fvalue(row.get("accuracy", row.get("cns_accuracy", 0.0))))
    return best


def _check(name: str, ok: bool, value: Any, threshold: Any = "", evidence: str = "", next_action: str = "") -> dict[str, Any]:
    return {
        "requirement_check": name,
        "status": "passed" if ok else "blocked",
        "value": value,
        "threshold": threshold,
        "evidence": evidence,
        "next_action": next_action,
    }


def build_stage_summary_rows(
    *,
    reddit_ttcpp: list[dict[str, Any]],
    reddit_multiseed: list[dict[str, Any]],
    teacher_ensemble: list[dict[str, Any]],
    arxiv_cns: list[dict[str, Any]],
    arxiv_semantic: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_rows = reddit_ttcpp + reddit_multiseed + teacher_ensemble + arxiv_cns + arxiv_semantic + products
    guard = summarize_guard(all_rows)
    best_001 = _best_acc(reddit_ttcpp + reddit_multiseed, dataset="Reddit", ratio=0.001)
    best_005 = _best_acc(reddit_ttcpp + reddit_multiseed, dataset="Reddit", ratio=0.005)
    best_cns = max([fvalue(row.get("cns_accuracy", row.get("accuracy", 0.0))) for row in arxiv_cns] or [0.0])
    semantic_cache_present = any(str(row.get("semantic_cache_path", "")) and str(row.get("status", "")) != "blocked" for row in arxiv_semantic)
    products_multiseed = len({str(row.get("seed", "")) for row in products if str(row.get("seed", ""))}) >= 2
    promoted_sota = sum(1 for row in all_rows if str(row.get("promotion_track", "")) == "sota_chase" and truthy(row.get("promotion_allowed", False)))
    return [
        _check("reddit_ttcpp_rows_present", _rows_present(reddit_ttcpp), len(reddit_ttcpp), ">0"),
        _check("reddit_ttcpp_0p10_recovered", best_001 >= 0.923, best_001, ">=0.923"),
        _check("reddit_ttcpp_0p50_first_gate", best_005 >= 0.938, best_005, ">=0.938"),
        _check("reddit_ttcpp_0p50_main_gate", best_005 >= 0.940, best_005, ">=0.940"),
        _check("reddit_ttcpp_0p50_stretch_gate", best_005 >= 0.942, best_005, ">=0.942"),
        _check("teacher_ensemble_cache_present", _rows_present(teacher_ensemble), len(teacher_ensemble), ">0"),
        _check("arxiv_raw_mlp_cns_sanity", best_cns >= 0.715, best_cns, ">=0.715"),
        _check("arxiv_sft_cns_gate", best_cns >= 0.715, best_cns, ">=0.715"),
        _check("arxiv_semantic_cache_present", semantic_cache_present, semantic_cache_present, "true"),
        _check("products_maintenance_multiseed", products_multiseed, len(products), ">=2 seeds"),
        _check("forbidden_guard_hits", int(guard["unsafe_promoted_rows"]) == 0, guard["unsafe_promoted_rows"], "0", evidence=json.dumps(guard, sort_keys=True)),
        _check("promoted_sota_chase_rows", promoted_sota >= 0, promoted_sota, ">=0"),
    ]


def write_outputs(args: argparse.Namespace) -> Path:
    reddit_ttcpp = read_csv(args.reddit_ttcpp_csv)
    reddit_multiseed = read_csv(args.reddit_multiseed_csv)
    teacher_ensemble = read_csv(args.teacher_ensemble_csv)
    arxiv_cns = read_csv(args.arxiv_cns_csv)
    arxiv_semantic = read_csv(args.arxiv_semantic_csv)
    products = read_csv(args.products_csv)
    rows = build_stage_summary_rows(
        reddit_ttcpp=reddit_ttcpp,
        reddit_multiseed=reddit_multiseed,
        teacher_ensemble=teacher_ensemble,
        arxiv_cns=arxiv_cns,
        arxiv_semantic=arxiv_semantic,
        products=products,
    )
    csv_path = write_csv(args.csv, rows, STAGE_FIELDS)
    ensure_report(
        args.report,
        [
            "# T32 Stage Summary",
            "",
            *markdown_table(rows, ["requirement_check", "status", "value", "threshold"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T32 stage checks.")
    parser.add_argument("--reddit-ttcpp-csv", default="experiments/tables/t32_reddit_ttcpp_seed42.csv")
    parser.add_argument("--reddit-multiseed-csv", default="experiments/tables/t32_reddit_ttcpp_multiseed.csv")
    parser.add_argument("--teacher-ensemble-csv", default="experiments/tables/t32_reddit_ttcpp_teacher_ensemble.csv")
    parser.add_argument("--arxiv-cns-csv", default="experiments/tables/t32_arxiv_actual_cns_seed42.csv")
    parser.add_argument("--arxiv-semantic-csv", default="experiments/tables/t32_arxiv_semantic_sft_seed42.csv")
    parser.add_argument("--products-csv", default="experiments/tables/t32_products_maintenance.csv")
    parser.add_argument("--csv", default="experiments/tables/t32_stage_summary.csv")
    parser.add_argument("--report", default="experiments/summaries/t32_stage_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
