from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.bonsai_sft_sketch import build_bonsai_sketch, lsh_bonsai_select
from shadow_hgc.sft.t29_contract import REDDIT_NUM_CLASSES, T29_REQUIRED_FIELDS, make_t29_row, ratio_budget


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_bonsai_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_reddit_bonsai.py --device cuda --ratios 0.001 0.005 "
        "--sketch-dim 64 128 --lsh-buckets 256 512 1024 --coverage-mode lsh reverse_knn_sampled "
        "--students table_head omcp weighted_sgc "
        f"--seed {int(seed)} --run-long"
    )


def build_bonsai_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", [0.001, 0.005])]
    sketch_dims = [int(v) for v in _arg(args, "sketch_dims", _arg(args, "sketch_dim", [64]))]
    lsh_buckets = [int(v) for v in _arg(args, "lsh_buckets", [256])]
    students = [str(v) for v in _arg(args, "students", ["table_head"])]
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        pool = max(budget * 2, budget + REDDIT_NUM_CLASSES)
        generator = torch.Generator().manual_seed(seed + budget)
        blocks = {
            "X0": torch.randn(pool, 12, generator=generator),
            "X1": torch.randn(pool, 12, generator=generator),
            "X2": torch.randn(pool, 12, generator=generator),
        }
        labels = torch.arange(pool) % REDDIT_NUM_CLASSES
        degree = torch.randint(1, 128, (pool,), generator=generator)
        for dim in sketch_dims:
            sketch = build_bonsai_sketch(blocks, labels=labels, degree=degree, output_dim=dim, seed=seed)
            for bucket_count in lsh_buckets:
                selection = lsh_bonsai_select(sketch.sketch, labels, total_budget=budget, lsh_buckets=bucket_count, seed=seed)
                for student in students:
                    rows.append(
                        make_t29_row(
                            dataset="Reddit",
                            method="reddit_sft_bonsai_sketch" if student == "table_head" else "reddit_sft_bonsai_omcp",
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            target_prototypes=budget,
                            status="completed_bonsai_smoke" if smoke else "completed_bonsai_selection_only",
                            promotion_status="not_promoted",
                            promotion_track="safe_mainline",
                            failure_reason="no_transfer_eval_accuracy",
                            notes="Bonsai sketch selection completed; no full pairwise coverage search.",
                            extra={
                                **sketch.diagnostics,
                                **selection.diagnostics,
                                "student_model": student,
                                "uses_exact_pairwise": False,
                            },
                        )
                    )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_bonsai_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_reddit_bonsai_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_reddit_bonsai_summary.md"),
        [
            "# T29 Reddit Bonsai Sketch",
            "",
            "- Bonsai sketch rows use approximate LSH coverage and avoid full pairwise search.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "actual_condensed_nodes", "student_model", "status", "uses_exact_pairwise", "accuracy", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_bonsai_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 Reddit Bonsai SFT sketch coverage.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--sketch-dim", "--sketch-dims", dest="sketch_dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--lsh-buckets", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--coverage-mode", nargs="+", default=["lsh"])
    parser.add_argument("--students", nargs="+", default=["table_head", "omcp", "weighted_sgc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_reddit_bonsai_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_reddit_bonsai_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
