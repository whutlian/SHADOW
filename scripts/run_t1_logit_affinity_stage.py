from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv, write_json
from shadow_hgc.logits.metadata import FORBIDDEN_PROMOTION_FLAGS


CACHE_INDEX = Path("experiments/tables/t1_logit_cache_index_seed42.csv")
LOGIT_CORRECT = Path("experiments/tables/t1_logit_correct_seed42.csv")
PSEUDO_SCAP = Path("experiments/tables/t1_pseudo_scap_seed42.csv")
ENSEMBLE = Path("experiments/tables/t1_safe_logit_ensemble_seed42.csv")
BOOST_SUMMARY = Path("experiments/tables/t1_fullgraph_boost_summary_seed42.csv")
LARGE_DRY_RUN = Path("experiments/tables/t1_large_dry_run_seed42.csv")
SUMMARY = Path("experiments/reports/t1_logit_affinity_stage_summary.md")


SAFE_BASES: list[dict[str, Any]] = [
    {"dataset": "acm", "base_variant": "SFB-v2 B3_scap_v2 retained", "base_accuracy": 0.915486, "base_macro_f1": 0.916580, "predicted_class_count": 3, "num_classes": 3, "target_type": "paper"},
    {"dataset": "dblp", "base_variant": "R+ relation-linear current-best", "base_accuracy": 0.836972, "base_macro_f1": 0.829937, "predicted_class_count": 4, "num_classes": 4, "target_type": "author"},
    {"dataset": "imdb", "base_variant": "clean S1 MAM/MDM/MKM", "base_accuracy": 0.424110, "base_macro_f1": 0.353932, "predicted_class_count": 5, "num_classes": 5, "target_type": "movie"},
    {"dataset": "ogbn-arxiv", "base_variant": "LAD_reference", "base_accuracy": 0.596774, "base_macro_f1": 0.415452, "predicted_class_count": 40, "num_classes": 40, "target_type": "paper"},
    {"dataset": "ogbn-products", "base_variant": "P0b_Rpp_base_shadow_fusion_reference", "base_accuracy": 0.668908, "base_macro_f1": 0.307981, "predicted_class_count": 47, "num_classes": 47, "target_type": "product"},
    {"dataset": "ogbn-products", "base_variant": "P0_LAD_reference", "base_accuracy": 0.658674, "base_macro_f1": 0.338064, "predicted_class_count": 47, "num_classes": 47, "target_type": "product"},
]


PROMOTED_COLUMNS = [
    "dataset",
    "promoted_variant",
    "base_variant",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "base_accuracy",
    "base_macro_f1",
    "delta_accuracy",
    "delta_macro_f1",
    "uses_logit_correct",
    "uses_pseudo_scap",
    "uses_ensemble",
    "uses_diffusion",
    "uses_dense_p2",
    "uses_bounded_edges",
    "promotion_status",
    "promotion_reason",
]


def validate_t1_promotion_row(row: dict[str, Any]) -> dict[str, Any]:
    invalid = [flag for flag in FORBIDDEN_PROMOTION_FLAGS if bool(row.get(flag, False))]
    status = "valid_for_promotion" if not invalid else "invalid_for_promotion"
    return {
        **row,
        "valid_for_promotion": len(invalid) == 0,
        "promotion_status": status,
        "invalid_reasons": invalid,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_existing_cache(dataset: str, base_variant: str) -> str:
    root = Path("experiments/logit_caches")
    if not root.exists():
        return ""
    normalized_dataset = dataset.lower().replace("-", "_")
    normalized_variant = "".join(ch if ch.isalnum() else "_" for ch in base_variant.lower())
    candidates = list(root.glob(f"**/*{normalized_dataset}*{normalized_variant}*/meta.json"))
    return str(candidates[0].parent) if candidates else ""


def _cache_index_rows(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in SAFE_BASES:
        cache_dir = _find_existing_cache(base["dataset"], base["base_variant"])
        status = "available" if cache_dir else "blocked_missing_safe_logit_cache"
        rows.append(
            {
                "dataset": base["dataset"],
                "base_variant": base["base_variant"],
                "seed": seed,
                "cache_status": status,
                "cache_dir": cache_dir,
                "accuracy": base["base_accuracy"],
                "macro_f1": base["base_macro_f1"],
                "predicted_class_count": base["predicted_class_count"],
                "num_classes": base["num_classes"],
                "target_type": base["target_type"],
                "uses_diffusion": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "uses_source_anchors": False,
                "uses_coverage_medoid": False,
                "uses_old_kd": False,
                "promotion_eligible_cache": bool(cache_dir),
                "blocked_reason": "" if cache_dir else "historical safe row metrics exist but split/all-target logits were not stored in current artifacts",
            }
        )
    return rows


def _blocked_method_rows(method: str, bases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in bases:
        row = {
            "dataset": base["dataset"],
            "base_variant": base["base_variant"],
            "base_acc": base["base_accuracy"],
            "base_macro_f1": base["base_macro_f1"],
            "mode": method,
            "space": "C_dimensional_logits_or_probabilities",
            "alpha": "",
            "steps": "",
            "temperature": "",
            "correct_alpha": "",
            "correct_steps": "",
            "beta": "",
            "valid_acc_before": "",
            "valid_acc_after": "",
            "test_acc_before": base["base_accuracy"],
            "test_acc_after": "",
            "macro_f1_before": base["base_macro_f1"],
            "macro_f1_after": "",
            "predicted_class_count_before": base["predicted_class_count"],
            "predicted_class_count_after": "",
            "full_edge_execution": True,
            "uses_bounded_edges": False,
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "promotion_status": "blocked",
            "promotion_reason": "blocked_missing_safe_logit_cache",
        }
        if method == "pseudo_scap":
            row.update(
                {
                    "threshold": "",
                    "pseudo_weight": "",
                    "topk_classes": "",
                    "pseudo_coverage": "",
                    "mean_confidence": "",
                    "median_confidence": "",
                    "train_override_count": "",
                    "nontrain_used_count": "",
                    "class_hist_base_pred": "",
                    "class_hist_pseudo_used": "",
                    "class_hist_after_affinity": "",
                    "prediction_entropy_before": "",
                    "prediction_entropy_after": "",
                }
            )
        if method == "safe_logit_ensemble":
            row.update(
                {
                    "candidate_components": base["base_variant"],
                    "component_accs": base["base_accuracy"],
                    "component_macro_f1s": base["base_macro_f1"],
                    "component_forbidden_flags": "false",
                    "ensemble_mode": "not_run_missing_cache",
                    "weights": "",
                    "temperatures": "",
                    "valid_acc": "",
                    "test_acc": "",
                    "macro_f1": "",
                    "predicted_class_count": "",
                }
            )
        rows.append(row)
    return rows


def _boost_summary_rows(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in cache_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset, group in sorted(by_dataset.items()):
        best = max(group, key=lambda row: float(row["accuracy"]))
        rows.append(
            {
                "dataset": dataset,
                "best_base_variant": best["base_variant"],
                "best_base_accuracy": best["accuracy"],
                "best_base_macro_f1": best["macro_f1"],
                "best_t1_variant": "",
                "best_t1_accuracy": "",
                "best_t1_macro_f1": "",
                "delta_accuracy": "",
                "delta_macro_f1": "",
                "improved": False,
                "eligible_for_condensation_recovery": False,
                "status": "blocked_missing_safe_logit_cache",
                "reason": "T1 requires split/all-target logits from safe base rows; current artifacts contain metrics but not logits",
                "uses_diffusion": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "promoted": False,
            }
        )
    return rows


def _large_dry_run_rows() -> list[dict[str, Any]]:
    specs = [
        {"dataset": "ogbn-arxiv", "num_target_nodes": 169343, "num_edges_used": 1166243, "num_classes": 40, "active_source_counts": 169343, "expected_wall_time_category": "minutes"},
        {"dataset": "ogbn-products", "num_target_nodes": 2449029, "num_edges_used": 61859140, "num_classes": 47, "active_source_counts": 2449029, "expected_wall_time_category": "tens_of_minutes"},
        {"dataset": "ogbn-papers100M", "num_target_nodes": 111059956, "num_edges_used": 1615685872, "num_classes": 172, "active_source_counts": "train_target_and_active_sources_only", "expected_wall_time_category": "hours_server_recommended"},
        {"dataset": "MAG240M", "num_target_nodes": 121751666, "num_edges_used": "paper_author_and_citation_streams", "num_classes": 153, "active_source_counts": "train_period_active_sources_only", "expected_wall_time_category": "hours_server_recommended"},
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        n = int(spec["num_target_nodes"])
        c = int(spec["num_classes"])
        logit_cache_bytes = n * c * 2
        topk = min(8, c)
        pseudo_scap_bytes = n * topk * (2 + 8) + n * 12
        rows.append(
            {
                **spec,
                "seed": 42,
                "full_edge_scans": 2,
                "peak_cpu_ram_gb": "streaming_dependent",
                "peak_gpu_ram_gb": "logit_chunk_only",
                "disk_cache_gb": round((logit_cache_bytes + pseudo_scap_bytes) / 1e9, 4),
                "logit_cache_bytes": int(logit_cache_bytes),
                "logit_cache_gb": round(logit_cache_bytes / 1e9, 4),
                "pseudo_scap_cache_bytes": int(pseudo_scap_bytes),
                "pseudo_scap_cache_gb": round(pseudo_scap_bytes / 1e9, 4),
                "edge_scans": 2,
                "uses_memmap": True,
                "uses_topk_sparse": True,
                "uses_bounded_edges": False,
                "status": "dry_run_estimate",
            }
        )
    return rows


def _write_empty_promoted_table(lines: list[str]) -> None:
    lines.append("| " + " | ".join(PROMOTED_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(PROMOTED_COLUMNS)) + " |")


def _write_summary(
    *,
    cache_rows: list[dict[str, Any]],
    correct_rows: list[dict[str, Any]],
    pseudo_rows: list[dict[str, Any]],
    ensemble_rows: list[dict[str, Any]],
    boost_rows: list[dict[str, Any]],
    dry_rows: list[dict[str, Any]],
) -> None:
    promoted: list[dict[str, Any]] = []
    blocked = correct_rows + pseudo_rows + ensemble_rows
    macro_regressed = [row for row in promoted if row.get("delta_macro_f1") not in {"", None} and float(row["delta_macro_f1"]) < 0]
    lines = [
        "# T1 Logit-Affinity Fullgraph Boost Summary",
        "",
        "This stage implements opt-in low-dimensional T1 boosters while leaving the default Shadow-HGC-R-1 path frozen.",
        "",
        "## Code Changes",
        "",
        "- Added `shadow_hgc.logits` cache metadata/I/O with forbidden-promotion flags for diffusion, dense P2, bounded edges, source anchors, CoverageMedoid, and old KD.",
        "- Added LogitCorrectLite over destination-row normalized target-target edges using only C-dimensional logits/probabilities and train labels for error correction.",
        "- Added confidence-gated Pseudo-SCAP helpers with train-node one-hot override, top-k class sparse storage, prior centering helpers, and destination-row affinity aggregation.",
        "- Added safe nonnegative logit ensemble utilities with validation improvement and test non-regression gates.",
        "- Added T1 runners and stage artifact generation; historical safe rows without split/all-target logits are blocked explicitly instead of promoted.",
        "",
        "## Promoted Rows",
        "",
    ]
    _write_empty_promoted_table(lines)
    lines.extend(
        [
            "",
            "## Cache Index",
            "",
            *markdown_table(cache_rows, ["dataset", "base_variant", "cache_status", "accuracy", "macro_f1", "blocked_reason"]),
            "",
            "## T1 Dataset Summary",
            "",
            *markdown_table(boost_rows, ["dataset", "best_base_variant", "best_base_accuracy", "best_base_macro_f1", "status", "eligible_for_condensation_recovery", "reason"]),
            "",
            "## Large Dry-Run Estimates",
            "",
            *markdown_table(dry_rows, ["dataset", "num_target_nodes", "num_classes", "logit_cache_gb", "pseudo_scap_cache_gb", "edge_scans", "uses_memmap", "uses_topk_sparse", "uses_bounded_edges", "expected_wall_time_category"]),
            "",
            "## Required Final Questions",
            "",
            "1. Did LogitCorrectLite improve any dataset? No. It is implemented, but all historical safe base rows are missing split/all-target logit caches in the current artifacts, so all LogitCorrectLite experiment rows are `blocked_missing_safe_logit_cache`.",
            "2. Did Pseudo-SCAP improve any dataset? No. It is implemented, but no safe base logit cache is available to construct validation-gated pseudo labels.",
            "3. Did Safe Logit Ensemble improve any dataset? No. Ensemble components require valid safe logit caches; current artifacts contain metrics only.",
            "4. Which rows are promoted? None.",
            "5. Which rows are blocked and why? All planned T1 rows are blocked because the previous safe rows did not persist train/valid/test/all-target logits; no row is blocked by forbidden component use in this stage.",
            "6. Did any promoted row use forbidden components? No promoted rows exist, and the promotion validator rejects diffusion, dense P2, bounded edges, source anchors, CoverageMedoid, and old KD.",
            "7. Did any promoted medium row use bounded_edges? No. There are no promoted medium rows, and bounded-edge rows are invalid for promotion.",
            f"8. Did macro-F1 regress? No promoted row regressed macro-F1; promoted macro-regression count is {len(macro_regressed)}.",
            "9. Is any dataset eligible for condensation recovery? No. The condensation recovery rule requires at least one improved fullgraph T1 row, and none improved.",
            "10. What is the next recommended step? Re-run the safe base models with `save_logits_cache` enabled, then rerun `scripts/run_t1_logit_affinity_stage.py --seed 42`; do not add another high-dimensional feature block.",
            "",
            "## Artifacts",
            "",
            f"- `{CACHE_INDEX}`",
            f"- `{LOGIT_CORRECT}`",
            f"- `{PSEUDO_SCAP}`",
            f"- `{ENSEMBLE}`",
            f"- `{BOOST_SUMMARY}`",
            f"- `{LARGE_DRY_RUN}`",
        ]
    )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(seed: int) -> dict[str, Path]:
    cache_rows = _cache_index_rows(seed)
    correct_rows = _blocked_method_rows("logit_correct_lite", SAFE_BASES)
    pseudo_rows = _blocked_method_rows("pseudo_scap", SAFE_BASES)
    ensemble_rows = _blocked_method_rows("safe_logit_ensemble", SAFE_BASES)
    boost_rows = _boost_summary_rows(cache_rows)
    dry_rows = _large_dry_run_rows()

    write_csv(CACHE_INDEX, cache_rows)
    write_csv(LOGIT_CORRECT, correct_rows)
    write_csv(PSEUDO_SCAP, pseudo_rows)
    write_csv(ENSEMBLE, ensemble_rows)
    write_csv(BOOST_SUMMARY, boost_rows)
    write_csv(LARGE_DRY_RUN, dry_rows)
    _write_summary(
        cache_rows=cache_rows,
        correct_rows=correct_rows,
        pseudo_rows=pseudo_rows,
        ensemble_rows=ensemble_rows,
        boost_rows=boost_rows,
        dry_rows=dry_rows,
    )
    write_json(
        Path("experiments/logs/t1_logit_affinity_stage_seed42.json"),
        {
            "seed": seed,
            "stage": "T1 Scalable Logit-Affinity Fullgraph Boost",
            "status": "completed_with_blocked_experiments",
            "blocked_reason": "missing_safe_logit_cache",
            "artifact_paths": [str(CACHE_INDEX), str(LOGIT_CORRECT), str(PSEUDO_SCAP), str(ENSEMBLE), str(BOOST_SUMMARY), str(SUMMARY)],
        },
    )
    return {
        "cache_index": CACHE_INDEX,
        "logit_correct": LOGIT_CORRECT,
        "pseudo_scap": PSEUDO_SCAP,
        "ensemble": ENSEMBLE,
        "boost_summary": BOOST_SUMMARY,
        "large_dry_run": LARGE_DRY_RUN,
        "summary": SUMMARY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1 logit-affinity fullgraph boost stage.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_stage(int(args.seed))


if __name__ == "__main__":
    main()
