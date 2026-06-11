from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.pseudo_label_transport import select_pltc_indices
from shadow_hgc.sft.t29_contract import REDDIT_NUM_CLASSES, T29_REQUIRED_FIELDS, make_t29_row, ratio_budget


DEFAULT_RATIOS = (0.001, 0.005)
DEFAULT_INITS = ("current_sft_signature_random", "sft_hnr_fdm_hybrid", "sft_bonsai_sketch")


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_pltc_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_reddit_pltc.py --device cuda --ratios 0.001 0.005 "
        "--teacher sft_fullgraph --prototype-inits current_sft_signature_random sft_hnr_fdm_hybrid sft_bonsai_sketch "
        "--pltc-modes random confidence_balanced uncertainty_balanced --combine-with-omcp "
        "--operator-topks 8 16 32 --students operator_sft_table_head weighted_sgc weighted_graphsage "
        "--hidden-dims 128 256 512 --epochs 60 120 200 --seed 42 --run-long"
    )


def _method(init: str, combine: bool) -> str:
    if init == "sft_bonsai_sketch" and combine:
        return "reddit_pltc_bonsai_omcp"
    if combine:
        return "reddit_pltc_omcp"
    if init == "sft_hnr_fdm_hybrid":
        return "reddit_pltc_hnr_hybrid"
    return "reddit_pltc_random"


def build_pltc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    inits = [str(v) for v in _arg(args, "prototype_inits", DEFAULT_INITS)]
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    combine = bool(_arg(args, "combine_with_omcp", False))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        generator = torch.Generator().manual_seed(seed + budget)
        teacher_probs = torch.softmax(torch.randn(max(budget * 2, budget + 1), REDDIT_NUM_CLASSES, generator=generator), dim=1)
        selection = select_pltc_indices(teacher_probs, total_budget=budget, seed=seed)
        for init in inits:
            rows.append(
                make_t29_row(
                    dataset="Reddit",
                    method=_method(init, combine),
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    target_prototypes=budget,
                    status="completed_pltc_smoke" if smoke else "blocked",
                    promotion_status="not_promoted",
                    promotion_track="sota_chase",
                    failure_reason="no_transfer_eval_accuracy" if smoke else "missing_teacher_predictions",
                    notes="PLTC selection diagnostics only; teacher soft labels are targets, not feature columns.",
                    extra={
                        **selection.diagnostics,
                        "pltc_num_hard_train_nodes": min(REDDIT_NUM_CLASSES, budget),
                        "uses_teacher_logits": True,
                        "uses_logits_as_input": False,
                        "uses_valid_labels_as_input": False,
                        "uses_test_labels_as_input": False,
                    },
                )
            )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_pltc_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_reddit_pltc_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_reddit_pltc_summary.md"),
        [
            "# T29 Reddit PLTC",
            "",
            "- PLTC is SOTA-chase only and logs `uses_teacher_logits=True`.",
            "- Smoke rows do not report transfer accuracy.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "actual_condensed_nodes", "promotion_track", "uses_teacher_logits", "status", "accuracy", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_pltc_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 Reddit pseudo-label transport condensation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--teacher", default="sft_fullgraph")
    parser.add_argument("--prototype-inits", nargs="+", default=list(DEFAULT_INITS))
    parser.add_argument("--pltc-modes", nargs="+", default=["random", "confidence_balanced"])
    parser.add_argument("--combine-with-omcp", action="store_true")
    parser.add_argument("--operator-topks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--students", nargs="+", default=["operator_sft_table_head", "weighted_sgc"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[60, 120, 200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_reddit_pltc_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_reddit_pltc_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
