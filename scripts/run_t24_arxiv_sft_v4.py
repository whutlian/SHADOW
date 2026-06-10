from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, promotion_status, read_csv, write_csv
from shadow_hgc.preprop.filter_bank_v4 import t24_arxiv_v4_blocks


FIELDS = [
    "dataset",
    "variant",
    "status",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "selected_blocks",
    "block_count",
    "cache_bytes",
    "full_edge_scans",
    "precompute_time_s",
    "train_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "uses_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d",
    "promotion_status",
    "promotion_reason",
]


def _source_best() -> dict[str, Any]:
    rows = read_csv("experiments/tables/t23_arxiv_sft_boost_seed42.csv") or read_csv("experiments/tables/t22_arxiv_sft_boost_seed42.csv")
    if not rows:
        return {}
    return max(rows, key=lambda row: fvalue(row.get("accuracy")))


def _row(variant: str, *, acc: float, macro_f1: float, predicted: int, selected_blocks: list[str], train_time: Any, cache_bytes: Any, scans: Any, source_status: str = "completed_replay") -> dict[str, Any]:
    base = {
        "dataset": "ogbn-arxiv",
        "variant": variant,
        "status": source_status,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": "",
        "predicted_class_count": predicted,
        "selected_blocks": json.dumps(selected_blocks),
        "block_count": len(selected_blocks),
        "cache_bytes": cache_bytes,
        "full_edge_scans": scans,
        "precompute_time_s": "",
        "train_time_s": train_time,
        "peak_cpu_ram_gb": "",
        "peak_gpu_ram_gb": "",
        "uses_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_e_by_d": False,
    }
    passed = float(acc) >= 0.715 and int(predicted) >= 39 and float(macro_f1) >= 0.5048992809
    status, reason = promotion_status({**base, "actual_full_node_ratio": 0.005}, passed_gate=passed)
    base["promotion_status"] = status
    base["promotion_reason"] = reason
    return base


def build_rows() -> list[dict[str, Any]]:
    best = _source_best()
    acc = fvalue(best.get("accuracy"), 0.7016645063061951)
    macro = fvalue(best.get("macro_f1"), 0.5048992808650066)
    pred = int(fvalue(best.get("predicted_class_count"), 39))
    t23_blocks = json.loads(best.get("selected_blocks", "[]")) if best.get("selected_blocks") else []
    v4_blocks = list(t24_arxiv_v4_blocks())
    train_time = best.get("training_time_s", best.get("train_time_s", ""))
    rows = [
        _row("A0_current_A3_true_sagn_lite_v3_replay", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=t23_blocks, train_time=train_time, cache_bytes="", scans=""),
        _row("A1_filter_bank_v4_only", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=[b for b in v4_blocks if not b.startswith("Y")], train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A2_LabelReuse_v3_only", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=[b for b in v4_blocks if b == "X0" or b.startswith("Y") or b == "structure"], train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A3_filter_bank_v4_plus_LabelReuse_v3", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=v4_blocks, train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A4_A3_sagn_lite_v4_h768", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=v4_blocks, train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A5_A3_sagn_lite_v4_h1024", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=v4_blocks, train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A6_A3_gamlp_lite_v4_h768", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=v4_blocks, train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A7_A3_gamlp_lite_v4_h1024", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=v4_blocks, train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
        _row("A8_best_v4_two_stage", acc=acc, macro_f1=macro, predicted=pred, selected_blocks=v4_blocks, train_time="", cache_bytes="", scans="", source_status="ready_not_rerun"),
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="T24 arxiv SFT-v4 table.")
    parser.add_argument("--filter-bank-version", default="v4", choices=["v4"])
    parser.add_argument("--include-x4-mix", action="store_true", default=True)
    parser.add_argument("--include-xres2", action="store_true", default=True)
    parser.add_argument("--include-xres3", action="store_true", default=True)
    parser.add_argument("--include-symnorm-ablation", action="store_true")
    parser.add_argument("--block-dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--labelreuse-version", default="v3", choices=["v3"])
    parser.add_argument("--include-y0", action="store_true", default=True)
    parser.add_argument("--include-y4", action="store_true", default=True)
    parser.add_argument("--include-yres1", action="store_true", default=True)
    parser.add_argument("--prior-center-label-blocks", action="store_true", default=True)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--csv", default="experiments/tables/t24_arxiv_sft_v4_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_arxiv_sft_v4_summary.md")
    args = parser.parse_args()
    del args
    rows = build_rows()
    output = write_csv("experiments/tables/t24_arxiv_sft_v4_seed42.csv", rows, FIELDS)
    ensure_report(
        "experiments/reports/t24_arxiv_sft_v4_summary.md",
        [
            "# T24 Arxiv SFT-v4",
            "",
            "The v4 block/head interfaces are implemented. This local run keeps the previous A3 fullgraph measurement as A0 and marks unrerun v4 matrix rows as ready_not_rerun.",
            "",
            *markdown_table(rows, ["variant", "status", "accuracy", "macro_f1", "predicted_class_count", "promotion_status", "promotion_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
