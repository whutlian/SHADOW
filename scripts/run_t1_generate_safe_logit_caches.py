from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.t1_safe_common import SAFE_BASES, save_cache_for_graph, train_sfb_v2_cache


FIELDS = [
    "dataset",
    "base_variant",
    "cache_status",
    "historical_test_acc",
    "replay_test_acc",
    "delta_replay",
    "macro_f1",
    "predicted_class_count",
    "train_nodes",
    "valid_nodes",
    "test_nodes",
    "all_target_nodes",
    "split_hash",
    "feature_hash",
    "model_config_hash",
    "cache_path",
    "gate_cache_path",
    "forbidden_component_flags",
    "blocked_reason",
]


def run(args) -> list[dict]:
    rows = []
    cache_root = Path(args.cache_root)
    for base in SAFE_BASES:
        if base["dataset"] == "acm" and base["cache_variant"] == "B3_scap_v2":
            graph, logits, train_rows, valid_rows, metrics = train_sfb_v2_cache(
                dataset="acm",
                cache_variant="B3_scap_v2",
                seed=args.seed,
                gate_mode=False,
                epochs=args.epochs,
            )
            replay_cache = save_cache_for_graph(
                root=cache_root,
                graph=graph,
                logits=logits,
                train_rows=train_rows,
                valid_rows=valid_rows,
                base=base,
                seed=args.seed,
                role="historical_replay",
                metrics=metrics,
                dtype=args.dtype,
            )
            gate_graph, gate_logits, gate_train, gate_valid, gate_metrics = train_sfb_v2_cache(
                dataset="acm",
                cache_variant="B3_scap_v2",
                seed=args.seed,
                gate_mode=True,
                epochs=args.epochs,
                val_fraction=args.val_fraction,
            )
            gate_cache = save_cache_for_graph(
                root=cache_root,
                graph=gate_graph,
                logits=gate_logits,
                train_rows=gate_train,
                valid_rows=gate_valid,
                base=base,
                seed=args.seed,
                role="gate_selection",
                metrics=gate_metrics,
                dtype=args.dtype,
            )
            rows.append(
                {
                    "dataset": base["dataset"],
                    "base_variant": base["base_variant"],
                    "cache_status": "available_unreplayed",
                    "historical_test_acc": base["expected_acc"],
                    "replay_test_acc": "",
                    "delta_replay": "",
                    "macro_f1": metrics["test"]["macro_f1"],
                    "predicted_class_count": metrics["test"]["predicted_class_count"],
                    "train_nodes": int(train_rows.numel()),
                    "valid_nodes": 0,
                    "test_nodes": int(graph.test_idx.numel()),
                    "all_target_nodes": int(graph.num_nodes[graph.target_type]),
                    "split_hash": json.loads((replay_cache / "metadata.json").read_text(encoding="utf-8"))["split_hash"],
                    "feature_hash": json.loads((replay_cache / "metadata.json").read_text(encoding="utf-8"))["feature_hash"],
                    "model_config_hash": json.loads((replay_cache / "metadata.json").read_text(encoding="utf-8"))["model_config_hash"],
                    "cache_path": str(replay_cache),
                    "gate_cache_path": str(gate_cache),
                    "forbidden_component_flags": "[]",
                    "blocked_reason": "",
                }
            )
        else:
            rows.append(
                {
                    "dataset": base["dataset"],
                    "base_variant": base["base_variant"],
                    "cache_status": "blocked_missing_replayable_logit_path",
                    "historical_test_acc": base["expected_acc"],
                    "replay_test_acc": "",
                    "delta_replay": "",
                    "macro_f1": base["macro_f1"],
                    "predicted_class_count": "",
                    "train_nodes": "",
                    "valid_nodes": "",
                    "test_nodes": "",
                    "all_target_nodes": "",
                    "split_hash": "",
                    "feature_hash": "",
                    "model_config_hash": "",
                    "cache_path": "",
                    "gate_cache_path": "",
                    "forbidden_component_flags": "[]",
                    "blocked_reason": "current historical safe-row script records metrics but does not expose replayable all-target logits",
                }
            )
    write_csv(args.output, rows, FIELDS)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# T1.1 Safe Logit Cache Summary\n\n"
        "ACM SFB-v2 B3 is regenerated with historical replay and validation-gate caches. Other historical safe rows remain blocked because their current scripts do not expose replayable all-target logits.\n\n"
        f"- CSV: `{args.output}`\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T1.1 safe-row logit caches.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--cache-root", default="experiments/logit_caches/t1_safe_seed42")
    parser.add_argument("--output", default="experiments/tables/t1_safe_logit_cache_index_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_safe_logit_cache_summary.md")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
