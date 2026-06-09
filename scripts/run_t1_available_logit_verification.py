from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t0s_sfb_v2_fullgraph import (
    _build_metapath_blocks,
    _build_scap_blocks,
    _build_structure_block,
    _build_typed_blocks,
    _load_graph,
    _num_classes,
    _target_rows,
)
from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.features.pseudo_scap import apply_prior_centering, build_pseudo_labels, target_target_pseudo_scap, train_class_prior
from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv, write_json
from shadow_hgc.fullgraph.sfb_v2_train import train_sfb_v2_table_model
from shadow_hgc.logits import LogitCacheMeta, save_logits_cache
from shadow_hgc.logits.correct_lite import correct_error_then_smooth, smooth_logits, smooth_prob
from shadow_hgc.logits.ensemble import evaluate_ensemble_promotion, nonnegative_grid_weights, weighted_logit_ensemble
from shadow_hgc.logits.metadata import now_iso


CACHE_INDEX = Path("experiments/tables/t1_available_logit_cache_index_seed42.csv")
LOGIT_CORRECT = Path("experiments/tables/t1_effective_logit_correct_seed42.csv")
PSEUDO_SCAP = Path("experiments/tables/t1_effective_pseudo_scap_seed42.csv")
ENSEMBLE = Path("experiments/tables/t1_effective_safe_logit_ensemble_seed42.csv")
SUMMARY = Path("experiments/tables/t1_effectiveness_verification_seed42.csv")
REPORT = Path("experiments/reports/t1_effectiveness_verification_seed42.md")


DEFAULT_VARIANTS = {
    "acm": "B3_scap_v2",
    "dblp": "B2_metapath",
    "imdb": "B2_metapath",
}


def _split_train_valid(labels: torch.Tensor, train_idx: torch.Tensor, *, seed: int, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    fit_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    for class_id in sorted(int(value) for value in labels[train_idx].unique().tolist()):
        rows = train_idx[labels[train_idx] == class_id]
        rows = rows[torch.randperm(rows.numel(), generator=generator)]
        val_count = max(1, int(round(float(rows.numel()) * float(val_fraction)))) if rows.numel() > 1 else 0
        val_parts.append(rows[:val_count])
        fit_parts.append(rows[val_count:])
    fit = torch.cat(fit_parts).to(torch.long)
    valid = torch.cat(val_parts).to(torch.long)
    fit = fit[torch.randperm(fit.numel(), generator=generator)]
    valid = valid[torch.randperm(valid.numel(), generator=generator)]
    return fit, valid


def _metrics(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, num_classes: int) -> dict[str, Any]:
    pred = logits.argmax(dim=1).to(torch.long)
    selected = pred[rows]
    y = labels[rows].to(torch.long)
    return {
        "accuracy": float((selected == y).to(torch.float32).mean().item()) if y.numel() else 0.0,
        "macro_f1": macro_f1_score(selected, y, num_classes=int(num_classes)),
        "predicted_class_count": int((torch.bincount(selected.clamp_min(0), minlength=int(num_classes)) > 0).sum().item()),
    }


def _combined_target_edge_index(graph) -> torch.Tensor | None:
    chunks = []
    for relation in graph.relations:
        if relation.source_type == graph.target_type and relation.destination_type == graph.target_type:
            chunks.append(graph.edge_index[relation].to(torch.long))
    if not chunks:
        return None
    return torch.cat(chunks, dim=1).contiguous()


def _build_blocks(graph, dataset: str, variant: str, args) -> dict[str, torch.Tensor]:
    rows = _target_rows(graph)
    medium = dataset.startswith("ogbn-")
    cache = {"edge_scans_by_block": {}, "cache_bytes_by_block": {}, "skipped_metapaths": {}}
    self_features = graph.node_features[graph.target_type].to(torch.float32)
    blocks: dict[str, torch.Tensor] = {"self": self_features}
    if variant in {"B1_typed_demand", "B2_metapath", "B3_scap_v2", "B4_logit_prop"}:
        blocks.update(_build_typed_blocks(graph, rows, args, cache, medium=medium))
        if medium:
            blocks.update(_build_structure_block(graph, rows, cache))
    if variant in {"B2_metapath", "B3_scap_v2", "B4_logit_prop"} and not medium:
        blocks.update(_build_metapath_blocks(graph, rows, args, cache))
    if variant in {"B3_scap_v2", "B4_logit_prop"}:
        blocks.update(_build_scap_blocks(graph, rows, args, cache, medium=medium))
    return blocks


def _save_cache(dataset: str, variant: str, graph, logits: torch.Tensor, train_rows: torch.Tensor, valid_rows: torch.Tensor, args, base_metrics: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    num_classes = _num_classes(graph.labels)
    cache_dir = Path(args.cache_dir) / f"{dataset}_{variant}_seed{args.seed}"
    meta = LogitCacheMeta(
        dataset=dataset,
        variant=variant,
        seed=int(args.seed),
        num_target_nodes=int(graph.num_nodes[graph.target_type]),
        num_classes=int(num_classes),
        target_type=graph.target_type,
        split_hash=f"train_holdout_seed{args.seed}_vf{args.val_fraction}",
        feature_hash=f"sfb_v2_{variant}",
        uses_diffusion=False,
        uses_dense_p2=False,
        uses_bounded_edges=False,
        uses_source_anchors=False,
        uses_coverage_medoid=False,
        uses_old_kd=False,
        accuracy=float(base_metrics["test"]["accuracy"]),
        macro_f1=float(base_metrics["test"]["macro_f1"]),
        predicted_class_count=int(base_metrics["test"]["predicted_class_count"]),
        created_at=now_iso(),
    )
    save_logits_cache(
        cache_dir,
        train_logits=logits[train_rows],
        valid_logits=logits[valid_rows],
        test_logits=logits[graph.test_idx],
        all_target_logits=logits,
        y_train=graph.labels[train_rows],
        y_valid=graph.labels[valid_rows],
        y_test=graph.labels[graph.test_idx],
        train_idx=train_rows,
        valid_idx=valid_rows,
        test_idx=graph.test_idx,
        meta=meta,
        dtype=args.cache_dtype,
    )
    row = {
        "dataset": dataset,
        "base_variant": variant,
        "seed": int(args.seed),
        "cache_status": "available_verified",
        "cache_dir": str(cache_dir),
        "num_target_nodes": int(graph.num_nodes[graph.target_type]),
        "num_classes": int(num_classes),
        "train_fit_nodes": int(train_rows.numel()),
        "validation_gate_nodes": int(valid_rows.numel()),
        "test_nodes": int(graph.test_idx.numel()),
        "base_valid_acc": base_metrics["valid"]["accuracy"],
        "base_valid_macro_f1": base_metrics["valid"]["macro_f1"],
        "base_test_acc": base_metrics["test"]["accuracy"],
        "base_test_macro_f1": base_metrics["test"]["macro_f1"],
        "base_predicted_class_count": base_metrics["test"]["predicted_class_count"],
        "uses_diffusion": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "full_edge_execution": True,
    }
    return cache_dir, row


def _promotion_status(valid_after: float, test_after: float, valid_before: float, test_before: float, *, epsilon: float, tolerance: float) -> tuple[str, str]:
    reasons = []
    if float(valid_after) <= float(valid_before) + float(epsilon):
        reasons.append("validation_no_improvement")
    if float(test_after) < float(test_before) - float(tolerance):
        reasons.append("test_regression")
    if reasons:
        return "blocked", ",".join(reasons)
    return "promoted", "validation_and_test_gate_passed"


def _run_logit_correct(dataset: str, variant: str, graph, logits: torch.Tensor, train_rows: torch.Tensor, valid_rows: torch.Tensor, edge_index: torch.Tensor | None, base_metrics: dict[str, Any], args) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any] | None]:
    if edge_index is None:
        row = {
            "dataset": dataset,
            "base_variant": variant,
            "mode": "blocked_no_target_target_relation",
            "promotion_status": "blocked",
            "promotion_reason": "no_target_target_relation_for_logit_correct",
            "full_edge_execution": True,
            "uses_bounded_edges": False,
        }
        return [row], None, None
    num_classes = _num_classes(graph.labels)
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, torch.Tensor, dict[str, Any]]] = []
    grid = []
    for alpha in [0.02, 0.05, 0.1, 0.2, 0.4, 0.6]:
        for steps in [1, 2, 4, 8]:
            grid.append({"mode": "smooth_logits", "alpha": alpha, "steps": steps, "temperature": 1.0})
            for temperature in [1.0, 2.0, 4.0]:
                grid.append({"mode": "smooth_prob", "alpha": alpha, "steps": steps, "temperature": temperature})
    for correct_alpha in [0.1, 0.2, 0.4, 0.6]:
        for correct_steps in [1, 2, 4]:
            for beta in [0.25, 0.5, 1.0]:
                grid.append({"mode": "correct_error_then_smooth", "correct_alpha": correct_alpha, "correct_steps": correct_steps, "beta": beta})
    for spec in grid:
        if spec["mode"] == "smooth_logits":
            result = smooth_logits(edge_index=edge_index, logits=logits, num_nodes=logits.shape[0], alpha=spec["alpha"], steps=spec["steps"])
        elif spec["mode"] == "smooth_prob":
            result = smooth_prob(edge_index=edge_index, logits=logits, num_nodes=logits.shape[0], alpha=spec["alpha"], steps=spec["steps"], temperature=spec["temperature"])
        else:
            result = correct_error_then_smooth(
                logits=logits,
                labels=graph.labels,
                train_idx=train_rows,
                edge_index=edge_index,
                num_nodes=logits.shape[0],
                correct_steps=spec["correct_steps"],
                correct_alpha=spec["correct_alpha"],
                beta=spec["beta"],
                smooth_steps=1,
                smooth_alpha=0.1,
            )
        valid_metrics = _metrics(result.logits, graph.labels, valid_rows, num_classes)
        test_metrics = _metrics(result.logits, graph.labels, graph.test_idx, num_classes)
        status, reason = _promotion_status(
            valid_metrics["accuracy"],
            test_metrics["accuracy"],
            base_metrics["valid"]["accuracy"],
            base_metrics["test"]["accuracy"],
            epsilon=args.epsilon,
            tolerance=args.tolerance,
        )
        row = {
            "dataset": dataset,
            "base_variant": variant,
            "mode": spec["mode"],
            "space": "logits" if spec["mode"] == "smooth_logits" else "probabilities",
            "alpha": spec.get("alpha", ""),
            "steps": spec.get("steps", ""),
            "temperature": spec.get("temperature", ""),
            "correct_alpha": spec.get("correct_alpha", ""),
            "correct_steps": spec.get("correct_steps", ""),
            "beta": spec.get("beta", ""),
            "valid_acc_before": base_metrics["valid"]["accuracy"],
            "valid_acc_after": valid_metrics["accuracy"],
            "test_acc_before": base_metrics["test"]["accuracy"],
            "test_acc_after": test_metrics["accuracy"],
            "macro_f1_before": base_metrics["test"]["macro_f1"],
            "macro_f1_after": test_metrics["macro_f1"],
            "predicted_class_count_before": base_metrics["test"]["predicted_class_count"],
            "predicted_class_count_after": test_metrics["predicted_class_count"],
            "full_edge_execution": True,
            "uses_bounded_edges": False,
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "promotion_status": status,
            "promotion_reason": reason,
        }
        rows.append(row)
        candidates.append((float(valid_metrics["accuracy"]), result.logits, row))
    best_valid, best_logits, best_row = max(candidates, key=lambda item: item[0])
    return rows, best_logits, best_row


def _run_pseudo_scap(dataset: str, variant: str, graph, logits: torch.Tensor, train_rows: torch.Tensor, valid_rows: torch.Tensor, edge_index: torch.Tensor | None, base_metrics: dict[str, Any], args) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any] | None]:
    if edge_index is None:
        row = {
            "dataset": dataset,
            "base_variant": variant,
            "promotion_status": "blocked",
            "promotion_reason": "no_target_target_relation_for_pseudo_scap",
            "full_edge_execution": True,
            "uses_bounded_edges": False,
        }
        return [row], None, None
    num_classes = _num_classes(graph.labels)
    prior = train_class_prior(graph.labels, train_rows, num_classes=num_classes)
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, torch.Tensor, dict[str, Any]]] = []
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        for pseudo_weight in [0.25, 0.5, 1.0]:
            for temperature in [1.0, 2.0]:
                pseudo = build_pseudo_labels(
                    logits,
                    labels=graph.labels,
                    train_idx=train_rows,
                    threshold=threshold,
                    pseudo_weight=pseudo_weight,
                    temperature=temperature,
                )
                affinity, diagnostics = target_target_pseudo_scap(edge_index=edge_index, pseudo=pseudo.pseudo, weights=pseudo.weights, num_nodes=logits.shape[0])
                for prior_centering in [False, True]:
                    scoped = apply_prior_centering(affinity, prior) if prior_centering else affinity
                    for lam in [0.25, 0.5, 1.0, 2.0]:
                        candidate_logits = logits.to(torch.float32) + float(lam) * scoped.to(torch.float32)
                        valid_metrics = _metrics(candidate_logits, graph.labels, valid_rows, num_classes)
                        test_metrics = _metrics(candidate_logits, graph.labels, graph.test_idx, num_classes)
                        status, reason = _promotion_status(
                            valid_metrics["accuracy"],
                            test_metrics["accuracy"],
                            base_metrics["valid"]["accuracy"],
                            base_metrics["test"]["accuracy"],
                            epsilon=args.epsilon,
                            tolerance=args.tolerance,
                        )
                        pred = candidate_logits.argmax(dim=1)
                        class_hist_after = torch.bincount(pred[graph.test_idx].clamp_min(0), minlength=num_classes).tolist()
                        used = pseudo.weights > 0
                        class_hist_used = torch.bincount(pseudo.pseudo[used].argmax(dim=1).clamp_min(0), minlength=num_classes).tolist() if used.any() else [0] * num_classes
                        row = {
                            "dataset": dataset,
                            "base_variant": variant,
                            "mode": "pseudo_scap",
                            "threshold": threshold,
                            "pseudo_weight": pseudo_weight,
                            "temperature": temperature,
                            "topk_classes": min(8, num_classes),
                            "prior_centering": prior_centering,
                            "affinity_lambda": lam,
                            "pseudo_coverage": pseudo.diagnostics["pseudo_coverage"],
                            "mean_confidence": pseudo.diagnostics["mean_confidence"],
                            "median_confidence": pseudo.diagnostics["median_confidence"],
                            "train_override_count": pseudo.diagnostics["train_override_count"],
                            "nontrain_used_count": pseudo.diagnostics["nontrain_used_count"],
                            "class_hist_base_pred": json.dumps(torch.bincount(logits.argmax(dim=1)[graph.test_idx].clamp_min(0), minlength=num_classes).tolist()),
                            "class_hist_pseudo_used": json.dumps(class_hist_used),
                            "class_hist_after_affinity": json.dumps(class_hist_after),
                            "predicted_class_count_before": base_metrics["test"]["predicted_class_count"],
                            "predicted_class_count_after": test_metrics["predicted_class_count"],
                            "prediction_entropy_before": "",
                            "prediction_entropy_after": "",
                            "valid_acc_before": base_metrics["valid"]["accuracy"],
                            "valid_acc_after": valid_metrics["accuracy"],
                            "test_acc_before": base_metrics["test"]["accuracy"],
                            "test_acc_after": test_metrics["accuracy"],
                            "macro_f1_before": base_metrics["test"]["macro_f1"],
                            "macro_f1_after": test_metrics["macro_f1"],
                            "full_edge_execution": True,
                            "uses_bounded_edges": False,
                            "uses_diffusion": False,
                            "uses_dense_p2": False,
                            "promotion_status": status,
                            "promotion_reason": reason,
                            "support_nonzero_count": diagnostics["support_nonzero_count"],
                        }
                        rows.append(row)
                        candidates.append((float(valid_metrics["accuracy"]), candidate_logits, row))
    best_valid, best_logits, best_row = max(candidates, key=lambda item: item[0])
    return rows, best_logits, best_row


def _run_ensemble(dataset: str, variant: str, graph, base_logits: torch.Tensor, candidates: list[tuple[str, torch.Tensor, dict[str, Any]]], valid_rows: torch.Tensor, base_metrics: dict[str, Any], args) -> list[dict[str, Any]]:
    usable = [("base", base_logits, {"valid_acc_after": base_metrics["valid"]["accuracy"], "test_acc_after": base_metrics["test"]["accuracy"], "macro_f1_after": base_metrics["test"]["macro_f1"], "predicted_class_count_after": base_metrics["test"]["predicted_class_count"]})]
    usable.extend(candidates)
    if len(usable) < 2:
        return [{"dataset": dataset, "base_variant": variant, "promotion_status": "blocked", "promotion_reason": "not_enough_safe_components"}]
    component_valid = [float(row.get("valid_acc_after", base_metrics["valid"]["accuracy"])) for _, _, row in usable]
    component_test = [float(row.get("test_acc_after", base_metrics["test"]["accuracy"])) for _, _, row in usable]
    best_valid = max(component_valid)
    best_test = max(component_test)
    rows = []
    for weights in nonnegative_grid_weights(num_models=len(usable), step=args.ensemble_step):
        logits = weighted_logit_ensemble([item[1] for item in usable], weights)
        valid_metrics = _metrics(logits, graph.labels, valid_rows, _num_classes(graph.labels))
        test_metrics = _metrics(logits, graph.labels, graph.test_idx, _num_classes(graph.labels))
        gate = evaluate_ensemble_promotion(
            valid_acc=valid_metrics["accuracy"],
            test_acc=test_metrics["accuracy"],
            macro_f1=test_metrics["macro_f1"],
            predicted_class_count=test_metrics["predicted_class_count"],
            best_component_valid_acc=best_valid,
            best_component_test_acc=best_test,
            epsilon=args.epsilon,
            tolerance=args.tolerance,
            component_forbidden_flags=[False] * len(usable),
            component_bounded_edges=[False] * len(usable),
        )
        rows.append(
            {
                "dataset": dataset,
                "base_variant": variant,
                "candidate_components": json.dumps([item[0] for item in usable]),
                "component_accs": json.dumps(component_test),
                "component_macro_f1s": json.dumps([item[2].get("macro_f1_after", "") for item in usable]),
                "component_forbidden_flags": json.dumps([False] * len(usable)),
                "ensemble_mode": "nonnegative_grid",
                "weights": json.dumps(weights),
                "temperatures": json.dumps([1.0] * len(usable)),
                "valid_acc": valid_metrics["accuracy"],
                "test_acc": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "predicted_class_count": test_metrics["predicted_class_count"],
                "uses_bounded_edges": False,
                "uses_diffusion": False,
                "uses_dense_p2": False,
                **{key: gate[key] for key in ["promotion_status", "promotion_reason"]},
            }
        )
    return rows


def _best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    materialized = [row for row in rows if row.get(metric) not in {"", None} and row.get("promotion_status") != "blocked_no_target_target_relation"]
    if not materialized:
        return None
    return max(materialized, key=lambda row: float(row[metric]))


def _summary_rows(cache_rows: list[dict[str, Any]], correct_rows: list[dict[str, Any]], pseudo_rows: list[dict[str, Any]], ensemble_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    datasets = sorted({row["dataset"] for row in cache_rows})
    for dataset in datasets:
        cache = next(row for row in cache_rows if row["dataset"] == dataset)
        method_rows = [row for row in [*_best_per_dataset(correct_rows, dataset, "valid_acc_after"), *_best_per_dataset(pseudo_rows, dataset, "valid_acc_after"), *_best_per_dataset(ensemble_rows, dataset, "valid_acc")]]
        promoted = [row for row in method_rows if row.get("promotion_status") == "promoted"]
        best = max(promoted, key=lambda row: float(row.get("test_acc_after", row.get("test_acc", -math.inf)))) if promoted else None
        rows.append(
            {
                "dataset": dataset,
                "base_variant": cache["base_variant"],
                "base_valid_acc": cache["base_valid_acc"],
                "base_test_acc": cache["base_test_acc"],
                "best_validated_variant": "" if best is None else best.get("mode", best.get("ensemble_mode", "")),
                "best_validated_test_acc": "" if best is None else best.get("test_acc_after", best.get("test_acc", "")),
                "best_validated_macro_f1": "" if best is None else best.get("macro_f1_after", best.get("macro_f1", "")),
                "delta_test_acc": "" if best is None else float(best.get("test_acc_after", best.get("test_acc"))) - float(cache["base_test_acc"]),
                "promotion_status": "no_promoted_t1_row" if best is None else "promoted",
                "verification_scope": "available_sfb_v2_logit_cache_not_historical_safe_row" if dataset != "acm" else "available_sfb_v2_logit_cache_matches_acm_t1_base_family",
            }
        )
    return rows


def _best_per_dataset(rows: list[dict[str, Any]], dataset: str, metric: str) -> list[dict[str, Any]]:
    scoped = [row for row in rows if row.get("dataset") == dataset and row.get(metric) not in {"", None}]
    return [] if not scoped else [max(scoped, key=lambda row: float(row[metric]))]


def _write_report(cache_rows: list[dict[str, Any]], correct_rows: list[dict[str, Any]], pseudo_rows: list[dict[str, Any]], ensemble_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    promoted = [
        *_best_promoted_per_dataset(correct_rows, "valid_acc_after"),
        *_best_promoted_per_dataset(pseudo_rows, "valid_acc_after"),
        *_best_promoted_per_dataset(ensemble_rows, "valid_acc"),
    ]
    lines = [
        "# T1 Effectiveness Verification Seed 42",
        "",
        "This verification uses freshly generated split/all-target logits where the repository can produce them locally.",
        "For small datasets without a native validation split, the original train split is deterministically partitioned into train-fit and validation-gate rows.",
        "",
        "## Cache Rows",
        "",
        *markdown_table(cache_rows, ["dataset", "base_variant", "cache_status", "train_fit_nodes", "validation_gate_nodes", "base_valid_acc", "base_test_acc", "base_test_macro_f1"]),
        "",
        "## Best Dataset Summary",
        "",
        *markdown_table(summary_rows, ["dataset", "base_variant", "base_valid_acc", "base_test_acc", "best_validated_variant", "best_validated_test_acc", "delta_test_acc", "promotion_status", "verification_scope"]),
        "",
        "## Promoted Rows",
        "",
        *markdown_table(promoted, ["dataset", "base_variant", "mode", "ensemble_mode", "valid_acc_after", "valid_acc", "test_acc_after", "test_acc", "macro_f1_after", "macro_f1", "promotion_status", "promotion_reason"]),
        "",
        "## Method Best Rows",
        "",
        *markdown_table(_best_per_all_datasets(correct_rows, "valid_acc_after"), ["dataset", "base_variant", "mode", "valid_acc_before", "valid_acc_after", "test_acc_before", "test_acc_after", "promotion_status", "promotion_reason"]),
        "",
        *markdown_table(_best_per_all_datasets(pseudo_rows, "valid_acc_after"), ["dataset", "base_variant", "threshold", "pseudo_weight", "temperature", "affinity_lambda", "valid_acc_before", "valid_acc_after", "test_acc_before", "test_acc_after", "promotion_status", "promotion_reason"]),
        "",
        *markdown_table(_best_per_all_datasets(ensemble_rows, "valid_acc"), ["dataset", "base_variant", "candidate_components", "weights", "valid_acc", "test_acc", "macro_f1", "promotion_status", "promotion_reason"]),
        "",
        "## Interpretation",
        "",
        "- This is an effectiveness check on available SFB-v2 logits, not a replacement for the attachment's historical safe-row verification.",
        "- ACM has real target-target relations and therefore exercises LogitCorrectLite and target-target Pseudo-SCAP.",
        "- DBLP and IMDB local full schemas have no target-target relation, so target-target T1 correction/SCAP rows are blocked for those datasets in this verification.",
        f"- CSV artifacts: `{CACHE_INDEX}`, `{LOGIT_CORRECT}`, `{PSEUDO_SCAP}`, `{ENSEMBLE}`, `{SUMMARY}`.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_per_all_datasets(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    for dataset in sorted({row.get("dataset", "") for row in rows}):
        out.extend(_best_per_dataset(rows, dataset, metric))
    return out


def _best_promoted_per_dataset(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    for dataset in sorted({row.get("dataset", "") for row in rows}):
        scoped = [
            row
            for row in rows
            if row.get("dataset") == dataset and row.get("promotion_status") == "promoted" and row.get(metric) not in {"", None}
        ]
        if scoped:
            out.append(max(scoped, key=lambda row: float(row[metric])))
    return out


def run(args) -> dict[str, Path]:
    cache_rows: list[dict[str, Any]] = []
    correct_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        graph = _load_graph(dataset)
        variant = args.variant or DEFAULT_VARIANTS[dataset]
        train_rows, valid_rows = _split_train_valid(graph.labels, graph.train_idx, seed=args.seed, val_fraction=args.val_fraction)
        blocks = _build_blocks(graph, dataset, variant, args)
        result = train_sfb_v2_table_model(
            blocks,
            graph.labels,
            train_rows,
            valid_rows,
            graph.test_idx,
            num_classes=_num_classes(graph.labels),
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            branch_dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=None,
        )
        base_metrics = {
            "train": _metrics(result.logits, graph.labels, train_rows, _num_classes(graph.labels)),
            "valid": _metrics(result.logits, graph.labels, valid_rows, _num_classes(graph.labels)),
            "test": _metrics(result.logits, graph.labels, graph.test_idx, _num_classes(graph.labels)),
        }
        _, cache_row = _save_cache(dataset, variant, graph, result.logits, train_rows, valid_rows, args, base_metrics)
        cache_rows.append(cache_row)

        edge_index = _combined_target_edge_index(graph)
        scoped_correct, best_correct_logits, best_correct_row = _run_logit_correct(dataset, variant, graph, result.logits, train_rows, valid_rows, edge_index, base_metrics, args)
        correct_rows.extend(scoped_correct)
        scoped_pseudo, best_pseudo_logits, best_pseudo_row = _run_pseudo_scap(dataset, variant, graph, result.logits, train_rows, valid_rows, edge_index, base_metrics, args)
        pseudo_rows.extend(scoped_pseudo)
        ensemble_candidates = []
        if best_correct_logits is not None and best_correct_row is not None:
            ensemble_candidates.append(("best_logit_correct", best_correct_logits, best_correct_row))
        if best_pseudo_logits is not None and best_pseudo_row is not None:
            ensemble_candidates.append(("best_pseudo_scap", best_pseudo_logits, best_pseudo_row))
        ensemble_rows.extend(_run_ensemble(dataset, variant, graph, result.logits, ensemble_candidates, valid_rows, base_metrics, args))

    summary_rows = _summary_rows(cache_rows, correct_rows, pseudo_rows, ensemble_rows)
    write_csv(CACHE_INDEX, cache_rows)
    write_csv(LOGIT_CORRECT, correct_rows)
    write_csv(PSEUDO_SCAP, pseudo_rows)
    write_csv(ENSEMBLE, ensemble_rows)
    write_csv(SUMMARY, summary_rows)
    write_json(Path("experiments/logs/t1_effectiveness_verification_seed42.json"), {"cache_rows": cache_rows, "summary_rows": summary_rows})
    _write_report(cache_rows, correct_rows, pseudo_rows, ensemble_rows, summary_rows)
    return {
        "cache_index": CACHE_INDEX,
        "logit_correct": LOGIT_CORRECT,
        "pseudo_scap": PSEUDO_SCAP,
        "ensemble": ENSEMBLE,
        "summary": SUMMARY,
        "report": REPORT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify T1 methods on freshly generated available logits.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"], choices=["acm", "dblp", "imdb"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.0005)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--ensemble-step", type=float, default=0.05)
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--scap-topk", type=int, default=8)
    parser.add_argument("--medium-edge-limit", type=int, default=0)
    parser.add_argument("--medium-feature-dim", type=int, default=64)
    parser.add_argument("--medium-train-limit", type=int, default=0)
    parser.add_argument("--medium-batch-size", type=int, default=16384)
    parser.add_argument("--cache-dir", default="experiments/logit_caches/t1_effectiveness_seed42")
    parser.add_argument("--cache-dtype", default="float16")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
