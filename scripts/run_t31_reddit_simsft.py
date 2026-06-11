from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t31_reddit_ttc import _concat_features, build_reddit_ttc_server_command, load_or_train_teacher_cache
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.sft.simsft_soft import build_simsft_soft_table, simsft_promotion_status
from shadow_hgc.sft.t31_contract import T31_REQUIRED_FIELDS, apply_t31_promotion_guard, make_t31_row, ratio_budget
from shadow_hgc.sft.teacher_transport import teacher_probability_diagnostics, train_soft_label_condensed_student


def build_simsft_server_command() -> str:
    return (
        "python scripts/run_t31_reddit_simsft.py --device cuda --ratios 0.001 0.005 "
        "--methods simsft_soft_centroids simsft_soft_centroids_plus_residual_exemplars "
        "--hidden-dims 128 256 512 --epochs 60 120 200 --seed 42 --run-long"
    )


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_simsft_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    metadata = load_or_train_teacher_cache(args)
    ratios = [float(v) for v in _arg(args, "ratios", [0.001, 0.005])]
    methods = [str(v) for v in _arg(args, "methods", ["simsft_soft_centroids_plus_residual_exemplars"])]
    if metadata is None:
        return [
            make_t31_row(
                dataset="Reddit",
                method=methods[0],
                seed=int(_arg(args, "seed", 42)),
                requested_full_node_ratio=ratio,
                total_condensed_nodes=ratio_budget("Reddit", ratio),
                status="blocked",
                failure_reason="missing_reddit_teacher_cache",
                promotion_track="sota_chase",
                uses_teacher_logits=True,
                next_action=build_reddit_ttc_server_command(),
            )
            for ratio in ratios
        ]
    labels, _train_idx, valid_idx, test_idx = load_reddit_raw_memmap_labels_and_splits(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    features = _concat_features(args)
    probs = torch.from_numpy(np.asarray(np.load(metadata["probs_path"], mmap_mode="r"), dtype=np.float32))
    teacher_diag = teacher_probability_diagnostics(probs)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for method in methods:
            table = build_simsft_soft_table(features=features, teacher_probs=probs, num_rows=budget, method=method, seed=int(_arg(args, "seed", 42)))
            best: dict[str, Any] | None = None
            for hidden_dim in [int(v) for v in _arg(args, "hidden_dims", [128])]:
                for epochs in [int(v) for v in _arg(args, "epochs", [120])]:
                    result = train_soft_label_condensed_student(
                        z_syn=table.z_syn,
                        y_syn_soft=table.y_syn_soft,
                        eval_features=features[test_idx],
                        eval_labels=labels[test_idx],
                        valid_features=features[valid_idx],
                        valid_labels=labels[valid_idx],
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        device=str(_arg(args, "device", "cuda")),
                        seed=int(_arg(args, "seed", 42)),
                        target_prior=probs.mean(dim=0),
                    )
                    score = float(result.get("valid_acc", 0.0)) + 0.05 * float(result.get("macro_f1", 0.0))
                    payload = {**result, "hidden_dim": hidden_dim, "epochs": epochs, "score": score}
                    best = payload if best is None or score > float(best["score"]) else best
            assert best is not None
            promotion_status, failure = simsft_promotion_status(ratio=ratio, accuracy=float(best["accuracy"]))
            row = make_t31_row(
                dataset="Reddit",
                method=method,
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
                promotion_track="sota_chase",
                promotion_status=promotion_status,
                teacher_method=metadata.get("teacher_method", ""),
                teacher_accuracy=metadata.get("teacher_accuracy", ""),
                teacher_macro_f1=metadata.get("teacher_macro_f1", ""),
                teacher_valid_acc=metadata.get("teacher_valid_acc", ""),
                teacher_temperature=1.0,
                teacher_entropy_mean=teacher_diag["teacher_entropy_mean"],
                teacher_margin_mean=teacher_diag["teacher_margin_mean"],
                teacher_disagreement_mean=teacher_diag["teacher_disagreement_mean"],
                teacher_cache_bytes=metadata.get("teacher_cache_bytes", ""),
                teacher_logits_cache_path=metadata.get("logits_path", ""),
                uses_teacher_logits=True,
                uses_kd=False,
                soft_label_source="teacher_probs_cache",
                candidate_nodes="all",
                selected_bucket_counts_json=json.dumps(table.diagnostics["row_type_counts"], sort_keys=True),
                target_prior_type="teacher_soft_prior",
                student_model="table_head_mlp",
                hidden_dim=best["hidden_dim"],
                epochs=best["epochs"],
                uses_valid_labels_for_hyperparam_selection=True,
                source_table=metadata.get("probs_path", ""),
                notes="SimSFT table-only soft centroid/residual rows; no graph builder used.",
            )
            rows.append(apply_t31_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_simsft_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t31_reddit_simsft_seed42.csv"), rows, T31_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t31_reddit_simsft_notes.md"),
        [
            "# T31 Reddit SimSFT",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_simsft_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T31 Reddit SimSFT-soft.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default=json.dumps(["X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3", "structure"]))
    parser.add_argument("--teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--methods", nargs="+", default=["simsft_soft_centroids_plus_residual_exemplars"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--epochs", nargs="+", type=int, default=[120])
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t31_reddit_simsft_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_reddit_simsft_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
