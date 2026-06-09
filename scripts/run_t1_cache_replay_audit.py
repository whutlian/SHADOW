from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.t1_safe_common import read_csv
from shadow_hgc.logits.replay import replay_logits_cache


FIELDS = [
    "dataset",
    "base_variant",
    "cache_status",
    "historical_test_acc",
    "replay_test_acc",
    "delta_replay",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "prediction_entropy",
    "train_nodes",
    "valid_nodes",
    "test_nodes",
    "all_target_nodes",
    "split_hash",
    "feature_hash",
    "cache_path",
    "gate_cache_path",
    "blocked_reason",
]


def run(args) -> list[dict]:
    rows = []
    for row in read_csv(args.cache_index):
        if row.get("cache_path"):
            replay = replay_logits_cache(row["cache_path"], historical_test_acc=float(row["historical_test_acc"]), tolerance=args.tolerance)
            replay["gate_cache_path"] = row.get("gate_cache_path", "")
            replay["blocked_reason"] = "" if replay["cache_status"] == "available_verified" else "replay accuracy mismatch"
            rows.append(replay)
        else:
            rows.append({**row, "weighted_f1": "", "prediction_entropy": ""})
    write_csv(args.output, rows, FIELDS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay T1.1 safe logit caches.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-index", default="experiments/tables/t1_safe_logit_cache_index_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t1_cache_replay_audit_seed42.csv")
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
