from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.stc import class_histogram_json, synthetic_class_count_stats
from shadow_hgc.sft.stc_contract import T27_REQUIRED_FIELDS, make_t27_row


REDDIT_NUM_NODES = 232_965
REDDIT_NUM_TRAIN = 153_431
REDDIT_NUM_CLASSES = 41
REDDIT_SFT_DIM = 602

REQUIRED_REDDIT_METHODS: tuple[str, ...] = (
    "reddit_random_frozen_init",
    "reddit_random_trainable_delta_rho005",
    "reddit_random_trainable_delta_rho010",
    "reddit_random_gm",
    "reddit_random_outer",
    "reddit_random_gm_plus_moment",
    "reddit_kcenter_trainable_delta",
    "reddit_medoid_trainable_delta",
)


def build_reddit_server_command(seeds: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5)) -> str:
    seed_text = " ".join(str(int(seed)) for seed in seeds)
    return (
        "python scripts/run_t27_stc_reddit.py --device cuda --ratios 0.005 0.01 "
        "--init current_sft_signature_random --methods frozen_init trainable_delta gradient_matching outer_loop "
        "--delta-rhos 0.05 0.10 "
        f"--seeds {seed_text}"
    )


def _method_config(method: str) -> dict[str, Any]:
    if method == "reddit_random_frozen_init":
        return {"init": "current_sft_signature_random", "objective": "frozen_init", "rho": "", "trainable_delta": False}
    if method == "reddit_random_trainable_delta_rho005":
        return {"init": "current_sft_signature_random", "objective": "trainable_delta", "rho": 0.05, "trainable_delta": True}
    if method == "reddit_random_trainable_delta_rho010":
        return {"init": "current_sft_signature_random", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_random_gm":
        return {"init": "current_sft_signature_random", "objective": "gradient_matching", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_random_outer":
        return {"init": "current_sft_signature_random", "objective": "outer_loop", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_random_gm_plus_moment":
        return {"init": "current_sft_signature_random", "objective": "gradient_matching", "rho": 0.10, "trainable_delta": True, "lambda_moment": 0.1}
    if method == "reddit_kcenter_trainable_delta":
        return {"init": "current_sft_signature_kcenter", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_medoid_trainable_delta":
        return {"init": "current_sft_signature_medoid", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    raise ValueError(f"unknown Reddit method: {method}")


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "completed_smoke" if args.smoke else "server_ready_not_run"
    failure = "local_smoke_not_full_reddit_run" if args.smoke else "server_command_required_for_reddit_seed_sweep"
    seeds = args.seeds if getattr(args, "seeds", None) else [args.seed]
    for seed in seeds:
        for ratio in args.ratios:
            syn_rows = max(REDDIT_NUM_CLASSES, int(round(float(ratio) * REDDIT_NUM_NODES)))
            labels = np.arange(syn_rows, dtype=np.int64) % REDDIT_NUM_CLASSES
            histogram = class_histogram_json(labels, num_classes=REDDIT_NUM_CLASSES)
            count_stats = synthetic_class_count_stats(labels, num_classes=REDDIT_NUM_CLASSES)
            for method in REQUIRED_REDDIT_METHODS:
                cfg = _method_config(method)
                rows.append(
                    make_t27_row(
                        dataset="Reddit",
                        method=method,
                        seed=int(seed),
                        requested_full_node_ratio=float(ratio),
                        original_num_nodes=REDDIT_NUM_NODES,
                        num_train_nodes=REDDIT_NUM_TRAIN,
                        num_classes=REDDIT_NUM_CLASSES,
                        syn_rows=syn_rows,
                        syn_feature_dim=REDDIT_SFT_DIM,
                        init_method=cfg["init"],
                        stc_objective=cfg["objective"],
                        stc_delta_rho=cfg["rho"],
                        trainable_delta=bool(cfg["trainable_delta"]),
                        outer_steps=int(args.stc_outer_steps) if cfg["objective"] != "frozen_init" else 0,
                        gm_num_heads=int(args.gm_num_heads) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                        gm_real_batch_size=int(args.gm_real_batch_size) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                        head_type=args.stc_head,
                        head_hidden_dim=int(args.stc_head_hidden_dim),
                        status=status,
                        failure_reason=failure,
                        notes="T27 Reddit row is honest smoke/server-ready output; full seed metrics require the listed server command.",
                        extra={
                            "lambda_moment": cfg.get("lambda_moment", 0.1),
                            "synthetic_class_histogram_json": histogram,
                            "selected_or_syn_class_count_min": count_stats["selected_or_syn_class_count_min"],
                            "selected_or_syn_class_count_median": count_stats["selected_or_syn_class_count_median"],
                            "selected_or_syn_class_count_max": count_stats["selected_or_syn_class_count_max"],
                            "total_time": round(time.perf_counter() - started, 6),
                        },
                    )
                )
    return rows


def write_reddit_outputs(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    csv_path = write_csv(args.csv, rows, T27_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T27 Reddit STC Notes",
            "",
            "- Required Reddit STC rows are declared at 0.50% and 1.00% full-node ratios.",
            "- Seed 42 smoke is local; seeds 1..5 are emitted by the server command and must not be inferred from seed 42.",
            "- No smoke row is promoted.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "seed", "method", "status", "stc_objective", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Full server command: `{build_reddit_server_command(seeds=[1, 2, 3, 4, 5])}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or declare T27 Reddit SFT-STC rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--init", default="current_sft_signature_random")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--delta-rhos", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--stc-outer-steps", type=int, default=1000)
    parser.add_argument("--gm-num-heads", type=int, default=1)
    parser.add_argument("--gm-real-batch-size", type=int, default=4096)
    parser.add_argument("--stc-head", default="hidden_mlp")
    parser.add_argument("--stc-head-hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t27_stc_reddit_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t27_reddit_stc_notes.md")
    args = parser.parse_args()
    csv_path = write_reddit_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
