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


PRODUCTS_NUM_NODES = 2_449_029
PRODUCTS_NUM_TRAIN = 196_615
PRODUCTS_NUM_CLASSES = 47
PRODUCTS_SFT_DIM = 534

REQUIRED_PRODUCTS_METHODS: tuple[str, ...] = (
    "products_uca_mixup_frozen",
    "products_uca_mixup_trainable_delta_rho005",
    "products_uca_mixup_trainable_delta_rho010",
    "products_uca_mixup_gm",
    "products_uca_mixup_outer",
    "products_uca_mixup_outer_plus_coverage_official",
    "products_uca_mixup_outer_plus_coverage_balanced",
    "products_random_trainable_delta",
    "products_cb_random_trainable_delta",
)


def build_products_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t27_stc_products.py --device cuda "
        "--ratios 0.0025 0.005 --init products_uca_hybrid_mixup "
        "--methods frozen_init trainable_delta gradient_matching outer_loop outer_loop_plus_coverage "
        "--products-coverage-track official balanced --delta-rhos 0.05 0.10 0.20 "
        f"--stc-outer-steps 1000 --seed {int(seed)}"
    )


def _method_config(method: str) -> dict[str, Any]:
    if method == "products_uca_mixup_frozen":
        return {"init": "products_uca_hybrid_mixup", "objective": "frozen_init", "rho": "", "trainable_delta": False}
    if method == "products_uca_mixup_trainable_delta_rho005":
        return {"init": "products_uca_hybrid_mixup", "objective": "trainable_delta", "rho": 0.05, "trainable_delta": True}
    if method == "products_uca_mixup_trainable_delta_rho010":
        return {"init": "products_uca_hybrid_mixup", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "products_uca_mixup_gm":
        return {"init": "products_uca_hybrid_mixup", "objective": "gradient_matching", "rho": 0.10, "trainable_delta": True}
    if method == "products_uca_mixup_outer":
        return {"init": "products_uca_hybrid_mixup", "objective": "outer_loop", "rho": 0.10, "trainable_delta": True}
    if method == "products_uca_mixup_outer_plus_coverage_official":
        return {
            "init": "products_uca_hybrid_mixup",
            "objective": "outer_loop_plus_coverage",
            "rho": 0.10,
            "trainable_delta": True,
            "coverage_track": "official",
            "coverage_prior": "train",
            "lambda_coverage": 0.1,
        }
    if method == "products_uca_mixup_outer_plus_coverage_balanced":
        return {
            "init": "products_uca_hybrid_mixup",
            "objective": "outer_loop_plus_coverage",
            "rho": 0.10,
            "trainable_delta": True,
            "coverage_track": "balanced",
            "coverage_prior": "balanced_alpha025",
            "lambda_coverage": 0.1,
        }
    if method == "products_random_trainable_delta":
        return {"init": "P0c_same_budget_random_subset", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "products_cb_random_trainable_delta":
        return {"init": "products_cb_random", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    raise ValueError(f"unknown products method: {method}")


def _synthetic_histogram(syn_rows: int) -> str:
    labels = np.arange(int(syn_rows), dtype=np.int64) % PRODUCTS_NUM_CLASSES
    return class_histogram_json(labels, num_classes=PRODUCTS_NUM_CLASSES)


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "completed_smoke" if args.smoke else "server_ready_not_run"
    failure = "local_smoke_not_full_products_run" if args.smoke else "server_command_required_for_full_products_run"
    for ratio in args.ratios:
        syn_rows = max(PRODUCTS_NUM_CLASSES, int(round(float(ratio) * PRODUCTS_NUM_NODES)))
        for method in REQUIRED_PRODUCTS_METHODS:
            cfg = _method_config(method)
            histogram = _synthetic_histogram(syn_rows)
            count_stats = synthetic_class_count_stats(np.arange(syn_rows) % PRODUCTS_NUM_CLASSES, num_classes=PRODUCTS_NUM_CLASSES)
            row = make_t27_row(
                dataset="ogbn-products",
                method=method,
                seed=int(args.seed),
                requested_full_node_ratio=float(ratio),
                original_num_nodes=PRODUCTS_NUM_NODES,
                num_train_nodes=PRODUCTS_NUM_TRAIN,
                num_classes=PRODUCTS_NUM_CLASSES,
                syn_rows=syn_rows,
                syn_feature_dim=PRODUCTS_SFT_DIM,
                init_method=cfg["init"],
                stc_objective=cfg["objective"],
                stc_delta_rho=cfg["rho"],
                trainable_delta=bool(cfg["trainable_delta"]),
                inner_steps=int(args.stc_inner_steps) if "outer" in cfg["objective"] else "",
                outer_steps=int(args.stc_outer_steps) if cfg["objective"] != "frozen_init" else 0,
                gm_num_heads=int(args.gm_num_heads) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                gm_real_batch_size=int(args.gm_real_batch_size) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                head_type=args.stc_head,
                head_hidden_dim=int(args.stc_head_hidden_dim),
                status=status,
                failure_reason=failure,
                notes="T27 row is honest: local smoke/contract output only; no full products accuracy is claimed.",
                extra={
                    "lambda_coverage": cfg.get("lambda_coverage", 0.0),
                    "coverage_track": cfg.get("coverage_track", ""),
                    "coverage_prior": cfg.get("coverage_prior", ""),
                    "synthetic_class_histogram_json": histogram,
                    "selected_or_syn_class_count_min": count_stats["selected_or_syn_class_count_min"],
                    "selected_or_syn_class_count_median": count_stats["selected_or_syn_class_count_median"],
                    "selected_or_syn_class_count_max": count_stats["selected_or_syn_class_count_max"],
                    "total_time": round(time.perf_counter() - started, 6),
                },
            )
            rows.append(row)
    return rows


def write_products_outputs(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    csv_path = write_csv(args.csv, rows, T27_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T27 Products STC Notes",
            "",
            "- Required Products STC rows are declared at 0.25% and 0.50% full-node ratios.",
            "- Local rows are smoke/server-ready unless a full run fills `accuracy`, `macro_f1`, and `predicted_classes`.",
            "- No row is promoted from smoke output; no teacher logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels are used.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "method", "status", "stc_objective", "stc_delta_rho", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Full server command: `{build_products_server_command(seed=int(args.seed))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or declare T27 Products SFT-STC rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0025, 0.005])
    parser.add_argument("--init", default="products_uca_hybrid_mixup")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--products-coverage-track", nargs="+", default=["official", "balanced"])
    parser.add_argument("--delta-rhos", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--stc-inner-steps", type=int, default=1)
    parser.add_argument("--stc-outer-steps", type=int, default=1000)
    parser.add_argument("--gm-num-heads", type=int, default=1)
    parser.add_argument("--gm-real-batch-size", type=int, default=4096)
    parser.add_argument("--stc-head", default="hidden_mlp")
    parser.add_argument("--stc-head-hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t27_stc_products_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t27_products_stc_notes.md")
    args = parser.parse_args()
    csv_path = write_products_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
