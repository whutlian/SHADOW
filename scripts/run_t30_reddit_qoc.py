from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.qoc_transfer_eval import train_qoc_table_head
from shadow_hgc.sft.quotient_operator import build_quotient_operator
from shadow_hgc.sft.t30_contract import T30_REQUIRED_FIELDS, make_t30_row, ratio_budget


DEFAULT_RATIOS = (0.001, 0.005)
DEFAULT_ASSIGNMENTS = (
    "qoc_class_conditional_online_kmeans",
    "qoc_sft_ctc_assignment",
    "qoc_sft_bonsai_assignment",
    "qoc_hybrid_assignment",
)
DEFAULT_TOPKS = (8, 16, 32)
DEFAULT_STUDENTS = ("operator_sft_table_head",)

CONTROL_REFERENCES = {
    0.0005: {"reddit_random_frozen_init": (0.8547654525, 0.7978628905)},
    0.001: {
        "current_sft_signature_random": (0.8983896738, 0.8433886103),
        "sft_hnr_fdm_hybrid": (0.9215841158, 0.8848907779),
        "reddit_random_frozen_init": (0.8983178644, 0.8428406957),
    },
    0.0025: {
        "current_sft_signature_random": (0.9163958853, 0.8805489216),
        "sft_hnr_fdm_hybrid": (0.9140441269, 0.8730428740),
    },
    0.005: {
        "current_sft_signature_random": (0.9244564925, 0.8862562818),
        "sft_hnr_fdm_hybrid": (0.9217097822, 0.8817167426),
        "reddit_random_frozen_init": (0.9212609734, 0.8824960392),
    },
    0.01: {
        "current_sft_signature_random": (0.9245283019, 0.8892093666),
        "sft_hnr_fdm_hybrid": (0.9236127318, 0.8881558607),
    },
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_reddit_qoc_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_reddit_qoc.py --device cuda --ratios 0.001 0.005 "
        "--assignment-modes qoc_class_conditional_online_kmeans qoc_sft_ctc_assignment qoc_sft_bonsai_assignment qoc_hybrid_assignment "
        "--operator-topks 8 16 32 --quotient-build-modes original_dest_normalized code_row_normalized_fallback "
        "--students operator_sft_table_head --hidden-dims 128 256 512 --epochs 60 120 200 "
        f"--seed {int(seed)} --run-long"
    )


def build_reddit_qoc_pltc_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_reddit_qoc.py --device cuda --ratios 0.001 0.005 "
        "--assignment-modes qoc_pltc_confidence_balanced qoc_pltc_uncertainty_balanced qoc_pltc_class_mass_balanced "
        "--teacher sft_fullgraph --enable-pltc --promotion-track sota_chase "
        "--operator-topks 8 16 32 --students operator_sft_table_head --hidden-dims 128 256 512 --epochs 60 120 200 "
        f"--seed {int(seed)} --run-long"
    )


def _method_for_mode(mode: str) -> str:
    mapping = {
        "qoc_class_conditional_online_kmeans": "reddit_qoc_hard_online_kmeans",
        "qoc_sft_ctc_assignment": "reddit_qoc_hard_ctc_assignment",
        "qoc_sft_bonsai_assignment": "reddit_qoc_hard_bonsai_assignment",
        "qoc_hybrid_assignment": "reddit_qoc_hard_hybrid_assignment",
        "qoc_pltc_confidence_balanced": "reddit_qoc_pltc_confidence_balanced",
        "qoc_pltc_uncertainty_balanced": "reddit_qoc_pltc_uncertainty_balanced",
        "qoc_pltc_class_mass_balanced": "reddit_qoc_pltc_class_mass_balanced",
    }
    return mapping.get(mode, f"reddit_{mode}")


def _track_for_mode(mode: str, explicit_track: str, enable_pltc: bool) -> str:
    if explicit_track:
        return explicit_track
    if enable_pltc or mode.startswith("qoc_pltc"):
        return "sota_chase"
    return "safe_main"


def _operator_smoke_diag(num_codewords: int, topk: int, mode: str) -> dict[str, Any]:
    edge_src = torch.arange(int(num_codewords), dtype=torch.long)
    edge_dst = (edge_src + 1) % int(num_codewords)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    assignments = torch.arange(int(num_codewords), dtype=torch.long)
    result = build_quotient_operator(edge_index=edge_index, assignments=assignments, num_codewords=int(num_codewords), topk=int(topk), mode=mode)
    return result.diagnostics


def _load_transfer_cache(cache_dir: str | Path) -> dict[str, torch.Tensor] | None:
    root = Path(cache_dir)
    required = ["input_syn.npy", "labels_syn.npy", "code_weights.npy", "input_real.npy", "labels_real.npy"]
    if not root.exists() or not all((root / name).exists() for name in required):
        return None
    return {
        "input_syn": torch.from_numpy(np.load(root / "input_syn.npy")).to(torch.float32),
        "labels_syn": torch.from_numpy(np.load(root / "labels_syn.npy")).to(torch.long),
        "code_weights": torch.from_numpy(np.load(root / "code_weights.npy")).to(torch.float32),
        "input_real": torch.from_numpy(np.load(root / "input_real.npy")).to(torch.float32),
        "labels_real": torch.from_numpy(np.load(root / "labels_real.npy")).to(torch.long),
    }


def build_reddit_control_rows(ratios: list[float], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        for method, (acc, macro) in CONTROL_REFERENCES.get(float(ratio), {}).items():
            budget = ratio_budget("Reddit", ratio)
            rows.append(
                make_t30_row(
                    dataset="Reddit",
                    method=method,
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    num_codewords=budget,
                    accuracy=acc,
                    macro_f1=macro,
                    predicted_classes=41,
                    status="completed_reference",
                    promotion_status="not_promoted",
                    promotion_track="safe_main",
                    failure_reason="control_reference_not_new_t30_method",
                    notes="T30 control reference carried from prior stages; not a QOC promotion candidate.",
                    transfer_eval_type="reference",
                    student_model="reference",
                )
            )
    return rows


def build_reddit_qoc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    modes = [str(v) for v in _arg(args, "assignment_modes", DEFAULT_ASSIGNMENTS)]
    topks = [int(v) for v in _arg(args, "operator_topks", DEFAULT_TOPKS)]
    quotient_modes = [str(v) for v in _arg(args, "quotient_build_modes", ["code_row_normalized_fallback"])]
    students = [str(v) for v in _arg(args, "students", DEFAULT_STUDENTS)]
    hidden_dims = [int(v) for v in _arg(args, "hidden_dims", [128])]
    epochs_list = [int(v) for v in _arg(args, "epochs", [60])]
    seed = int(_arg(args, "seed", 42))
    explicit_track = str(_arg(args, "promotion_track", ""))
    cache = _load_transfer_cache(_arg(args, "sft_cache_dir", ""))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for mode in modes:
            track = _track_for_mode(mode, explicit_track, bool(_arg(args, "enable_pltc", False)))
            labeled = int(round(budget * (0.5 if track == "sota_chase" else 0.7)))
            for qmode in quotient_modes:
                for topk in topks:
                    diag_started = time.perf_counter()
                    diag = _operator_smoke_diag(budget, topk, qmode)
                    diag["operator_build_time"] = float(time.perf_counter() - diag_started)
                    for student in students:
                        base_kwargs = dict(
                            dataset="Reddit",
                            method=_method_for_mode(mode),
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            num_codewords=budget,
                            num_labeled_codewords=labeled,
                            num_unlabeled_codewords=max(0, budget - labeled),
                            total_condensed_edges=int(diag["operator_edges_after_topk"]),
                            promotion_status="not_promoted",
                            promotion_track=track,
                            assignment_mode=mode,
                            operator_mode="sparse_codeword_quotient",
                            quotient_build_mode=qmode,
                            student_model=student,
                            extra=diag,
                            uses_teacher_logits=track == "sota_chase",
                        )
                        if cache is None:
                            rows.append(
                                make_t30_row(
                                    **base_kwargs,
                                    status="completed_operator_smoke" if bool(_arg(args, "smoke", False)) else "blocked",
                                    failure_reason="no_transfer_eval_accuracy" if bool(_arg(args, "smoke", False)) else "missing_reddit_sft_transfer_cache",
                                    transfer_eval_type="operator_smoke",
                                    notes="QOC operator path ran, but real transfer cache is missing so no accuracy is reported.",
                                    next_action=build_reddit_qoc_server_command(seed),
                                )
                            )
                            continue
                        result = train_qoc_table_head(
                            input_syn=cache["input_syn"],
                            labels_syn=cache["labels_syn"],
                            code_weights=cache["code_weights"],
                            input_real=cache["input_real"],
                            labels_real=cache["labels_real"],
                            num_classes=41,
                            hidden_dim=hidden_dims[0],
                            epochs=epochs_list[0],
                            seed=seed,
                        )
                        rows.append(
                            make_t30_row(
                                **base_kwargs,
                                status="completed_transfer_eval",
                                failure_reason="",
                                accuracy=result.metrics["accuracy"],
                                macro_f1=result.metrics["macro_f1"],
                                predicted_classes=result.metrics["predicted_classes"],
                                transfer_eval_type="real_transfer_eval",
                                extra={**diag, **result.metrics},
                                source_table=str(Path(_arg(args, "sft_cache_dir", ""))),
                            )
                        )
                    if bool(_arg(args, "smoke", False)):
                        break
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    seeds_arg = _arg(args, "seeds", None)
    seeds = [int(v) for v in (seeds_arg if seeds_arg else [int(_arg(args, "seed", 42))])]
    rows: list[dict[str, Any]] = []
    original_seed = int(_arg(args, "seed", 42))
    for seed in seeds:
        args.seed = seed
        if seed == 42:
            rows.extend(build_reddit_control_rows(ratios, seed))
        rows.extend(build_reddit_qoc_rows(args))
    args.seed = original_seed
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t30_reddit_qoc_seed42.csv"), rows, T30_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t30_reddit_qoc_notes.md"),
        [
            "# T30 Reddit QOC",
            "",
            "- Control references are carried separately from QOC rows.",
            "- QOC rows require real transfer cache before accuracy can be reported.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "num_codewords", "operator_topk", "operator_row_sum_error", "transfer_eval_type", "accuracy", "macro_f1", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- QOC-hard command: `{build_reddit_qoc_server_command(seed=int(_arg(args, 'seed', 42)))}`",
            f"- QOC-soft command: `{build_reddit_qoc_pltc_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T30 Reddit Shadow-QOC experiments.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--assignment-modes", nargs="+", default=list(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--operator-topks", nargs="+", type=int, default=list(DEFAULT_TOPKS))
    parser.add_argument("--quotient-build-modes", nargs="+", default=["code_row_normalized_fallback"])
    parser.add_argument("--students", nargs="+", default=list(DEFAULT_STUDENTS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[60, 120, 200])
    parser.add_argument("--teacher", default="")
    parser.add_argument("--enable-pltc", action="store_true")
    parser.add_argument("--promotion-track", default="")
    parser.add_argument("--sft-cache-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t30_reddit_qoc_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t30_reddit_qoc_notes.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
