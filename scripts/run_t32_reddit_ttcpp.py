from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from scripts.run_t31_reddit_ttc import DEFAULT_BLOCKS, _concat_features, _selected_blocks, _train_blockwise_soft_student, load_or_train_teacher_cache
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.t32_contract import T32_REQUIRED_FIELDS, apply_t32_promotion_guard, make_t32_row, ratio_budget, ttcpp_promotion_status
from shadow_hgc.sft.teacher_transport import TTCCondensedTable
from shadow_hgc.sft.ttcpp_progressive import add_allnode_repair_rows, score_ttc_rows_for_progressive_compression, select_progressive_subset
from shadow_hgc.sft.ttcpp_selector import compute_selected_soft_prior, select_ttc_rows_ratio_adaptive, soft_prior_kl


METHODS = [
    "reddit_ttcpp_ratio_adaptive_core70",
    "reddit_ttcpp_ratio_adaptive_core55",
    "reddit_ttcpp_ratio_adaptive_core40",
    "reddit_ttcpp_teacher_ensemble_confidence",
    "reddit_ttcpp_teacher_ensemble_disagreement",
    "reddit_ttcpp_teacher_ensemble_coverage_boundary",
    "reddit_ttcpp_progressive_0p50_to_0p10",
    "reddit_ttcpp_progressive_0p50_to_0p25",
    "reddit_ttcpp_progressive_with_20pct_repair",
    "reddit_ttcpp_calibrated_mixup",
    "reddit_ttcpp_residual_gated_student",
    "reddit_ttcpp_sagn_table_student",
    "reddit_ttcpp_gamlp_table_student",
    "reddit_ttcpp_swa_ema",
]


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_reddit_ttcpp_server_command() -> str:
    return (
        "python scripts/run_t32_reddit_ttcpp.py --device cuda --ratios 0.001 0.0025 0.005 "
        "--methods reddit_ttcpp_ratio_adaptive_core70 reddit_ttcpp_ratio_adaptive_core55 "
        "reddit_ttcpp_ratio_adaptive_core40 reddit_ttcpp_teacher_ensemble_confidence "
        "reddit_ttcpp_teacher_ensemble_disagreement reddit_ttcpp_teacher_ensemble_coverage_boundary "
        "reddit_ttcpp_progressive_0p50_to_0p10 reddit_ttcpp_progressive_0p50_to_0p25 "
        "reddit_ttcpp_progressive_with_20pct_repair reddit_ttcpp_calibrated_mixup "
        "reddit_ttcpp_sagn_table_student reddit_ttcpp_gamlp_table_student reddit_ttcpp_swa_ema "
        "--hidden-dims 128 256 --epochs 120 200 --temperatures 2 4 --lambda-hard 0.25 0.5 "
        "--lambda-prior 0.02 0.05 --seed 42 --run-long"
    )


def _load_teacher_probs(args: argparse.Namespace) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    ensemble_dir = Path(_arg(args, "teacher_ensemble_cache_dir", "experiments/cache/t32_reddit_teacher_ensemble_seed42"))
    ensemble_probs = ensemble_dir / "teacher_probs.npy"
    ensemble_disagreement = ensemble_dir / "teacher_disagreement.npy"
    if ensemble_probs.exists():
        probs = torch.from_numpy(np.asarray(np.load(ensemble_probs, mmap_mode="r"), dtype=np.float32))
        disagreement = None
        if ensemble_disagreement.exists():
            disagreement = torch.from_numpy(np.asarray(np.load(ensemble_disagreement, mmap_mode="r"), dtype=np.float32))
        meta_path = ensemble_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        diag = dict(meta.get("diagnostics", {}))
        diag["probs_path"] = str(ensemble_probs)
        diag["teacher_cache_bytes"] = int(ensemble_probs.stat().st_size + (ensemble_disagreement.stat().st_size if ensemble_disagreement.exists() else 0))
        return probs, disagreement, diag
    metadata = load_or_train_teacher_cache(args)
    if metadata is None:
        return None, None, {}
    probs = torch.from_numpy(np.asarray(np.load(metadata["probs_path"], mmap_mode="r"), dtype=np.float32))
    return probs, None, metadata


def _method_policy(method: str) -> str:
    if "core70" in method:
        return "ratio_adaptive_core70"
    if "core55" in method:
        return "ratio_adaptive_core55"
    if "core40" in method:
        return "ratio_adaptive_core40"
    if "disagreement" in method:
        return "teacher_ensemble_disagreement"
    if "coverage_boundary" in method:
        return "teacher_ensemble_coverage_boundary"
    if "confidence" in method:
        return "teacher_ensemble_confidence"
    if "calibrated_mixup" in method:
        return "calibrated_mixup_core40"
    return "ratio_adaptive_core40"


def _table_from_node_ids(
    *,
    features: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    node_ids: torch.Tensor,
    row_type: str,
) -> TTCCondensedTable:
    ids = node_ids.detach().cpu().long()
    z = features[ids].float().cpu()
    y_soft = probs[ids].float().cpu()
    train_set = {int(v) for v in train_idx.detach().cpu().tolist()}
    hard_mask = torch.tensor([int(v) in train_set for v in ids.tolist()], dtype=torch.bool)
    y_hard = torch.full((ids.numel(),), -1, dtype=torch.long)
    if hard_mask.any():
        y_hard[hard_mask] = labels.detach().cpu().long()[ids[hard_mask]]
    prior_kl = soft_prior_kl(compute_selected_soft_prior(y_soft), probs.mean(dim=0))
    diagnostics = {
        "condensed_nodes": int(ids.numel()),
        "row_type_counts_json": json.dumps({row_type: int(ids.numel())}, sort_keys=True),
        "selected_soft_prior_kl": prior_kl,
        "entropy_bucket_coverage": "",
        "margin_bucket_coverage": "",
        "disagreement_bucket_coverage": "",
        "class_coverage_min": "",
        "class_coverage_median": "",
        "class_coverage_max": "",
        "mixup_virtual_count": 0,
    }
    return TTCCondensedTable(
        z_syn=z,
        y_syn_soft=y_soft,
        y_syn_hard=y_hard,
        hard_anchor_mask=hard_mask,
        source_node_ids=ids,
        bucket_types=[row_type] * int(ids.numel()),
        sample_weight=torch.ones(int(ids.numel()), dtype=torch.float32),
        diagnostics=diagnostics,
    )


def _build_table_for_method(
    *,
    method: str,
    ratio: float,
    budget: int,
    args: argparse.Namespace,
    features: torch.Tensor,
    probs: torch.Tensor,
    disagreement: torch.Tensor | None,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> TTCCondensedTable:
    if "progressive" not in method:
        return select_ttc_rows_ratio_adaptive(
            features=features,
            teacher_probs=probs,
            labels=labels,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            num_rows=budget,
            ratio=ratio,
            policy=_method_policy(method),
            seed=int(_arg(args, "seed", 42)),
            disagreement=disagreement,
            mixup_alpha=float(_arg(args, "mixup_alpha", 0.4)),
        )
    source_ratio = 0.005
    source_budget = max(budget, ratio_budget("Reddit", source_ratio))
    source = select_ttc_rows_ratio_adaptive(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=source_budget,
        ratio=source_ratio,
        policy="ratio_adaptive_core40",
        seed=int(_arg(args, "seed", 42)),
        disagreement=disagreement,
    )
    entropy = -(source.y_syn_soft.clamp_min(1e-12) * source.y_syn_soft.clamp_min(1e-12).log()).sum(dim=1)
    top2 = torch.topk(source.y_syn_soft, k=min(2, source.y_syn_soft.shape[1]), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    scores = score_ttc_rows_for_progressive_compression(source.y_syn_soft, teacher_prior=probs.mean(dim=0), entropy=entropy, margin=margin)
    subset = select_progressive_subset(scores, target_budget=budget)
    node_ids = source.source_node_ids[subset]
    node_ids = node_ids[node_ids >= 0]
    repair_fraction = 0.20 if "20pct_repair" in method else 0.0
    if repair_fraction > 0.0 or node_ids.numel() < budget:
        node_ids = add_allnode_repair_rows(
            node_ids,
            probs[node_ids] if node_ids.numel() else probs[:0],
            probs,
            teacher_prior=probs.mean(dim=0),
            target_budget=budget,
            repair_fraction=max(repair_fraction, 0.10),
        )
    return _table_from_node_ids(features=features, probs=probs, labels=labels, train_idx=train_idx, node_ids=node_ids[:budget], row_type="progressive")


def _args_for_method(args: argparse.Namespace, method: str) -> argparse.Namespace:
    cloned = copy.copy(args)
    if "gamlp" in method:
        setattr(cloned, "student_model_type", "gamlp_lite")
    else:
        setattr(cloned, "student_model_type", _arg(args, "student_model_type", "sagn_lite_v4"))
    return cloned


def _blocked_rows(args: argparse.Namespace, ratios: list[float], methods: list[str], reason: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        for method in methods:
            rows.append(
                make_t32_row(
                    dataset="Reddit",
                    method=method,
                    seed=int(_arg(args, "seed", 42)),
                    requested_full_node_ratio=ratio,
                    condensed_nodes=ratio_budget("Reddit", ratio),
                    status="blocked",
                    failure_reason=reason,
                    promotion_track="sota_chase",
                    promotion_status="not_promoted",
                    uses_teacher_logits=True,
                    uses_kd=True,
                    uses_logits_as_input=False,
                    next_action=build_reddit_ttcpp_server_command(),
                )
            )
    return rows


def build_reddit_ttcpp_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", [0.001, 0.005])]
    methods = [str(v) for v in _arg(args, "methods", ["reddit_ttcpp_ratio_adaptive_core40"])]
    probs, disagreement, teacher_meta = _load_teacher_probs(args)
    if probs is None:
        return _blocked_rows(args, ratios, methods, "missing_reddit_teacher_cache")
    labels, train_idx, valid_idx, test_idx = load_reddit_raw_memmap_labels_and_splits(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    features = _concat_features(args)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for method in methods:
            started = time.perf_counter()
            table = _build_table_for_method(
                method=method,
                ratio=ratio,
                budget=budget,
                args=args,
                features=features,
                probs=probs,
                disagreement=disagreement,
                labels=labels,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
            )
            best: dict[str, Any] | None = None
            for hidden_dim in [int(v) for v in _arg(args, "hidden_dims", [128])]:
                for epochs in [int(v) for v in _arg(args, "epochs", [120])]:
                    for temperature in [float(v) for v in _arg(args, "temperatures", [2.0])]:
                        for lambda_hard in [float(v) for v in _arg(args, "lambda_hard", [0.5])]:
                            for lambda_prior in [float(v) for v in _arg(args, "lambda_prior", [0.05])]:
                                result = _train_blockwise_soft_student(
                                    args=_args_for_method(args, method),
                                    table=table,
                                    labels=labels,
                                    valid_idx=valid_idx,
                                    test_idx=test_idx,
                                    hidden_dim=hidden_dim,
                                    epochs=epochs,
                                    temperature=temperature,
                                    lambda_hard=lambda_hard,
                                    lambda_prior=lambda_prior,
                                    target_prior=probs.mean(dim=0),
                                )
                                score = float(result.get("valid_acc", 0.0))
                                candidate = {
                                    **result,
                                    "hidden_dim": hidden_dim,
                                    "epochs": epochs,
                                    "temperature": temperature,
                                    "lambda_hard": lambda_hard,
                                    "lambda_prior": lambda_prior,
                                    "score": score,
                                }
                                best = candidate if best is None or score > float(best["score"]) else best
            assert best is not None
            promotion_status, gate_reason = ttcpp_promotion_status(ratio=ratio, accuracy=float(best["accuracy"]), macro_f1=float(best["macro_f1"]))
            diag = table.diagnostics
            row = make_t32_row(
                dataset="Reddit",
                method=method,
                seed=int(_arg(args, "seed", 42)),
                requested_full_node_ratio=ratio,
                condensed_nodes=budget,
                shadow_nodes=0,
                condensed_edges=0,
                accuracy=best["accuracy"],
                macro_f1=best["macro_f1"],
                valid_acc=best["valid_acc"],
                predicted_classes=best["predicted_classes"],
                status="completed_long",
                failure_reason=gate_reason,
                promotion_track="sota_chase",
                promotion_status=promotion_status,
                uses_teacher_logits=True,
                uses_kd=True,
                uses_logits_as_input=False,
                candidate_nodes="all",
                student_model=_args_for_method(args, method).student_model_type,
                hidden_dim=best["hidden_dim"],
                epochs=best["epochs"],
                dropout=float(_arg(args, "dropout", 0.1)),
                weight_decay=float(_arg(args, "weight_decay", 5e-4)),
                lambda_soft=1.0,
                lambda_hard=best["lambda_hard"],
                lambda_prior=best["lambda_prior"],
                lambda_conf=float(_arg(args, "lambda_conf", 0.0)),
                lambda_mix=float(_arg(args, "lambda_mix", 0.0)),
                soft_temperature=best["temperature"],
                budget_policy=diag.get("budget_policy", _method_policy(method)),
                row_type_counts_json=diag.get("row_type_counts_json", ""),
                selected_soft_prior_kl=diag.get("selected_soft_prior_kl", ""),
                entropy_bucket_coverage=diag.get("entropy_bucket_coverage", ""),
                margin_bucket_coverage=diag.get("margin_bucket_coverage", ""),
                disagreement_bucket_coverage=diag.get("disagreement_bucket_coverage", ""),
                class_coverage_min=diag.get("class_coverage_min", ""),
                class_coverage_median=diag.get("class_coverage_median", ""),
                class_coverage_max=diag.get("class_coverage_max", ""),
                teacher_accuracy=teacher_meta.get("teacher_accuracy", teacher_meta.get("accuracy", "")),
                teacher_valid_acc=teacher_meta.get("teacher_valid_acc", ""),
                teacher_temperature=teacher_meta.get("teacher_temperature", ""),
                teacher_entropy_mean=teacher_meta.get("teacher_entropy_mean", ""),
                teacher_disagreement_mean=teacher_meta.get("teacher_disagreement_mean", ""),
                cache_bytes=teacher_meta.get("teacher_cache_bytes", teacher_meta.get("cache_bytes", "")),
                selection_time=float(time.perf_counter() - started),
                training_time="included_in_selection_time",
                peak_cpu_ram=current_cpu_ram_bytes(),
                peak_gpu_ram=current_gpu_ram_bytes(),
                notes="T32 TTC++ uses teacher soft targets over all nodes; logits/probs are not input features; hyperparameter selection uses validation accuracy only.",
                next_action=build_reddit_ttcpp_server_command(),
            )
            rows.append(apply_t32_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_reddit_ttcpp_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t32_reddit_ttcpp_seed42.csv"), rows, T32_REQUIRED_FIELDS)
    if str(_arg(args, "multiseed_csv", "")):
        write_csv(_arg(args, "multiseed_csv", "experiments/tables/t32_reddit_ttcpp_multiseed.csv"), rows, T32_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t32_reddit_ttcpp_summary.md"),
        [
            "# T32 Reddit TTC++",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_reddit_ttcpp_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T32 Reddit TTC++ ratio-adaptive and progressive experiments.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default=json.dumps(DEFAULT_BLOCKS))
    parser.add_argument("--teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--teacher-ensemble-cache-dir", default="experiments/cache/t32_reddit_teacher_ensemble_seed42")
    parser.add_argument("--teacher-model-type", default="sagn_lite_v4")
    parser.add_argument("--teacher-hidden-dim", type=int, default=128)
    parser.add_argument("--teacher-dropout", type=float, default=0.3)
    parser.add_argument("--teacher-num-layers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--methods", nargs="+", default=["reddit_ttcpp_ratio_adaptive_core40"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[2.0])
    parser.add_argument("--lambda-hard", nargs="+", type=float, default=[0.5])
    parser.add_argument("--lambda-prior", nargs="+", type=float, default=[0.05])
    parser.add_argument("--lambda-conf", type=float, default=0.0)
    parser.add_argument("--lambda-mix", type=float, default=0.0)
    parser.add_argument("--student-model-type", default="sagn_lite_v4")
    parser.add_argument("--student-lr", type=float, default=0.003)
    parser.add_argument("--student-batch-size", type=int, default=2048)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--epochs", nargs="+", type=int, default=[120])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t32_reddit_ttcpp_seed42.csv")
    parser.add_argument("--multiseed-csv", default="experiments/tables/t32_reddit_ttcpp_multiseed.csv")
    parser.add_argument("--report", default="experiments/summaries/t32_reddit_ttcpp_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
