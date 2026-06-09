from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.t1_safe_common import read_csv


FIELDS = ["dataset", "base_variant", "path_set", "alpha", "space", "valid_acc_before", "valid_acc_after", "test_acc_before", "test_acc_after", "promotion_status", "promotion_reason", "uses_dense_path_adjacency", "exposes_metapath_edge_type"]


def run(args) -> list[dict]:
    rows = []
    for row in read_csv(args.replay_audit):
        if row["dataset"] in {"dblp", "imdb"} and row.get("cache_status") == "available_verified":
            rows.append({**_blocked(row, "path_logit_correct_not_wired_for_this_cache"), "path_set": ""})
        elif row["dataset"] in {"dblp", "imdb"}:
            rows.append({**_blocked(row, row.get("blocked_reason") or row.get("cache_status", "cache_not_verified")), "path_set": ""})
    if not rows:
        rows.append(_blocked({"dataset": "dblp", "base_variant": "R+ relation-linear current-best"}, "cache_not_verified"))
        rows.append(_blocked({"dataset": "imdb", "base_variant": "clean S1 MAM/MDM/MKM"}, "cache_not_verified"))
    write_csv(args.output, rows, FIELDS)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# T1.1 PathLogitCorrectLite Summary\n\nDBLP/IMDB historical safe caches were not replay-verified, so path correction is blocked.\n\n" + f"- CSV: `{args.output}`\n", encoding="utf-8")
    return rows


def _blocked(row: dict, reason: str) -> dict:
    return {
        "dataset": row.get("dataset", ""),
        "base_variant": row.get("base_variant", ""),
        "alpha": "",
        "space": "",
        "valid_acc_before": "",
        "valid_acc_after": "",
        "test_acc_before": "",
        "test_acc_after": "",
        "promotion_status": "blocked",
        "promotion_reason": reason,
        "uses_dense_path_adjacency": False,
        "exposes_metapath_edge_type": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1.1 safe PathLogitCorrectLite.")
    parser.add_argument("--replay-audit", default="experiments/tables/t1_cache_replay_audit_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t1_path_logit_correct_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_path_logit_correct_summary.md")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
