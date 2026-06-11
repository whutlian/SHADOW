from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t31_reddit_ttc import _concat_features, build_reddit_ttc_server_command, load_or_train_teacher_cache
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.sft.bonsai_sft_coverage import select_bonsai_coverage
from shadow_hgc.sft.t31_contract import T31_REQUIRED_FIELDS, apply_t31_promotion_guard, make_t31_row, ratio_budget
from shadow_hgc.sft.teacher_transport import train_soft_label_condensed_student


def build_bonsai_server_command() -> str:
    return (
        "python scripts/run_t31_reddit_bonsai_coverage.py --device cuda --ratios 0.001 0.005 "
        "--modes hard_train_label_coverage soft_ttc_coverage coverage_plus_boundary --hidden-dims 128 256 512 "
        "--epochs 60 120 --seed 42 --run-long"
    )


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _promotion_status(mode: str, ratio: float, accuracy: float) -> tuple[str, str]:
    if mode == "hard_train_label_coverage":
        return "not_promoted", "bonsai_coverage_reference_only"
    if abs(float(ratio) - 0.001) < 1e-12 and float(accuracy) >= 0.925:
        return "promoted", ""
    if abs(float(ratio) - 0.005) < 1e-12 and float(accuracy) >= 0.930:
        return "promoted", ""
    return "not_promoted", "bonsai_ttc_gate_not_met"


def build_bonsai_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_idx, valid_idx, test_idx = load_reddit_raw_memmap_labels_and_splits(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    features = _concat_features(args)
    metadata = load_or_train_teacher_cache(args)
    probs = None
    if metadata is not None:
        probs = torch.from_numpy(np.asarray(np.load(metadata["probs_path"], mmap_mode="r"), dtype=np.float32))
    ratios = [float(v) for v in _arg(args, "ratios", [0.001, 0.005])]
    modes = [str(v) for v in _arg(args, "modes", ["hard_train_label_coverage", "soft_ttc_coverage"])]
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for mode in modes:
            if mode != "hard_train_label_coverage" and probs is None:
                rows.append(
                    make_t31_row(
                        dataset="Reddit",
                        method=f"bonsai_{mode}",
                        seed=int(_arg(args, "seed", 42)),
                        requested_full_node_ratio=ratio,
                        total_condensed_nodes=budget,
                        status="blocked",
                        failure_reason="missing_reddit_teacher_cache",
                        promotion_track="sota_chase",
                        uses_teacher_logits=True,
                        next_action=build_reddit_ttc_server_command(),
                    )
                )
                continue
            result = select_bonsai_coverage(
                features=features,
                labels=labels,
                train_idx=train_idx,
                num_rows=budget,
                mode=mode,
                teacher_probs=probs,
                seed=int(_arg(args, "seed", 42)),
            )
            selected = result.selected_idx
            if mode == "hard_train_label_coverage":
                y_soft = F.one_hot(labels[selected], num_classes=int(labels.max().item()) + 1).float()
                hard = labels[selected]
                hard_mask = torch.ones(selected.numel(), dtype=torch.bool)
            else:
                assert probs is not None
                y_soft = probs[selected]
                hard = torch.full((selected.numel(),), -1, dtype=torch.long)
                hard_mask = torch.zeros(selected.numel(), dtype=torch.bool)
            best: dict[str, Any] | None = None
            for hidden_dim in [int(v) for v in _arg(args, "hidden_dims", [128])]:
                for epochs in [int(v) for v in _arg(args, "epochs", [120])]:
                    metric = train_soft_label_condensed_student(
                        z_syn=features[selected],
                        y_syn_soft=y_soft,
                        train_anchor_hard=hard,
                        hard_anchor_mask=hard_mask,
                        eval_features=features[test_idx],
                        eval_labels=labels[test_idx],
                        valid_features=features[valid_idx],
                        valid_labels=labels[valid_idx],
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        device=str(_arg(args, "device", "cuda")),
                        seed=int(_arg(args, "seed", 42)),
                        target_prior=y_soft.mean(dim=0),
                    )
                    score = float(metric.get("valid_acc", 0.0)) + 0.05 * float(metric.get("macro_f1", 0.0))
                    payload = {**metric, "hidden_dim": hidden_dim, "epochs": epochs, "score": score}
                    best = payload if best is None or score > float(best["score"]) else best
            assert best is not None
            promotion_status, failure = _promotion_status(mode, ratio, float(best["accuracy"]))
            row = make_t31_row(
                dataset="Reddit",
                method=f"bonsai_{mode}",
                seed=int(_arg(args, "seed", 42)),
                requested_full_node_ratio=ratio,
                total_condensed_nodes=budget,
                syn_rows=budget,
                accuracy=best["accuracy"],
                macro_f1=best["macro_f1"],
                valid_acc=best["valid_acc"],
                predicted_classes=best["predicted_classes"],
                status="completed_long",
                failure_reason=failure,
                promotion_track=result.diagnostics["promotion_track"],
                promotion_status=promotion_status,
                uses_teacher_logits=result.diagnostics["uses_teacher_logits"],
                uses_kd=False,
                candidate_nodes=result.diagnostics["candidate_nodes"],
                selected_bucket_counts_json=json.dumps({"lsh_selected": int(selected.numel())}, sort_keys=True),
                student_model="table_head_mlp",
                hidden_dim=best["hidden_dim"],
                epochs=best["epochs"],
                uses_valid_labels_for_hyperparam_selection=True,
                notes=json.dumps(result.diagnostics, sort_keys=True),
                source_table=metadata.get("probs_path", "") if metadata else "",
            )
            rows.append(apply_t31_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_bonsai_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t31_reddit_bonsai_coverage_seed42.csv"), rows, T31_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t31_reddit_bonsai_coverage_notes.md"),
        [
            "# T31 Reddit Bonsai Coverage",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "promotion_track", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_bonsai_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T31 Reddit Bonsai coverage selector.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default=json.dumps(["X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3", "structure"]))
    parser.add_argument("--teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--modes", nargs="+", default=["hard_train_label_coverage", "soft_ttc_coverage"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--epochs", nargs="+", type=int, default=[120])
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t31_reddit_bonsai_coverage_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_reddit_bonsai_coverage_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
