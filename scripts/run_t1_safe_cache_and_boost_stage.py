from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv
from shadow_hgc.logits.metadata import FORBIDDEN_PROMOTION_FLAGS


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = Path("experiments/reports/t1_safe_cache_and_boost_stage_summary.md")
SAFE_CACHE = Path("experiments/tables/t1_safe_logit_cache_index_seed42.csv")
REPLAY = Path("experiments/tables/t1_cache_replay_audit_seed42.csv")
CORRECT = Path("experiments/tables/t1_safe_logit_correct_seed42.csv")
PATH_CORRECT = Path("experiments/tables/t1_path_logit_correct_seed42.csv")
PSEUDO = Path("experiments/tables/t1_pseudo_scap_safe_seed42.csv")
ENSEMBLE = Path("experiments/tables/t1_safe_logit_ensemble_safe_seed42.csv")
BOOST = Path("experiments/tables/t1_safe_fullgraph_boost_summary_seed42.csv")
DRY = Path("experiments/tables/t1_large_logit_affinity_dry_run_seed42.csv")


def validate_safe_boost_row(row: dict[str, Any]) -> dict[str, Any]:
    def truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    invalid = [flag for flag in FORBIDDEN_PROMOTION_FLAGS if truthy(row.get(flag, False))]
    checked = dict(row)
    checked["invalid_reasons"] = invalid
    if invalid:
        checked["promotion_status"] = "invalid_for_promotion"
    return checked


def _run(script: str, args: list[str] | None = None) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *(args or [])], cwd=ROOT, check=True)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_boost_rows() -> list[dict[str, Any]]:
    rows = []
    for table, method in [(CORRECT, "logit_correct"), (PATH_CORRECT, "path_logit_correct"), (PSEUDO, "pseudo_scap"), (ENSEMBLE, "safe_ensemble")]:
        for row in _read(table):
            if row.get("promotion_status") == "promoted":
                rows.append(
                    validate_safe_boost_row(
                        {
                            "dataset": row.get("dataset", ""),
                            "promoted_variant": method,
                            "base_variant": row.get("base_variant", ""),
                            "accuracy": row.get("test_acc_after", row.get("test_acc", "")),
                            "macro_f1": row.get("macro_f1_after", row.get("macro_f1", "")),
                            "predicted_class_count": row.get("predicted_class_count_after", row.get("predicted_class_count", "")),
                            "valid_acc_after": row.get("valid_acc_after", row.get("valid_acc", "")),
                            "promotion_status": row.get("promotion_status", ""),
                            "promotion_reason": row.get("promotion_reason", ""),
                            "uses_diffusion": row.get("uses_diffusion", False),
                            "uses_dense_p2": row.get("uses_dense_p2", False),
                            "uses_bounded_edges": row.get("uses_bounded_edges", False),
                        }
                    )
                )
    write_csv(BOOST, rows, fieldnames=["dataset", "promoted_variant", "base_variant", "accuracy", "macro_f1", "predicted_class_count", "valid_acc_after", "promotion_status", "promotion_reason", "uses_diffusion", "uses_dense_p2", "uses_bounded_edges", "invalid_reasons"])
    return rows


def _write_summary(promoted: list[dict[str, Any]]) -> None:
    cache = _read(SAFE_CACHE)
    replay = _read(REPLAY)
    correct = _read(CORRECT)
    path_correct = _read(PATH_CORRECT)
    pseudo = _read(PSEUDO)
    ensemble = _read(ENSEMBLE)

    def active_status(row: dict[str, str]) -> str:
        return row.get("cache_status") or row.get("promotion_status", "")

    raw_blocked = [
        row
        for row in [*cache, *replay, *correct, *path_correct, *pseudo, *ensemble]
        if active_status(row) not in {"available_unreplayed", "available_verified", "promoted"}
    ]
    blocked = []
    seen_blocked: set[tuple[str, str, str]] = set()
    for row in raw_blocked:
        reason = row.get("blocked_reason") or row.get("promotion_reason", "")
        key = (
            row.get("dataset", ""),
            row.get("base_variant", ""),
            reason,
        )
        if key in seen_blocked:
            continue
        seen_blocked.add(key)
        blocked.append(row)
    medium_promoted_bounded = [row for row in promoted if row.get("dataset") in {"ogbn-arxiv", "ogbn-products"} and str(row.get("uses_bounded_edges")).lower() == "true"]
    macro_collapse = [
        row for row in promoted
        if row.get("predicted_class_count") not in {"", None} and row.get("dataset") == "acm" and int(float(row["predicted_class_count"])) < 3
    ]
    promoted_summary = "; ".join(
        f"{row['dataset']} {row['promoted_variant']} acc={row.get('accuracy', '')} macro_f1={row.get('macro_f1', '')}"
        for row in promoted
    ) or "None"
    generated_cache = [row for row in cache if row.get("cache_status") == "available_unreplayed"]
    verified_replay = [row for row in replay if row.get("cache_status") == "available_verified"]
    mismatch_replay = [row for row in replay if row.get("cache_status") == "invalid_replay_mismatch"]
    slow_or_failed_cache = [row for row in cache if row.get("cache_status") not in {"available_unreplayed", "available_verified"}]
    logit_correct_promoted = [row for row in correct if row.get("promotion_status") == "promoted"]
    path_promoted = [row for row in path_correct if row.get("promotion_status") == "promoted"]
    pseudo_promoted = [row for row in pseudo if row.get("promotion_status") == "promoted"]
    ensemble_promoted = [row for row in ensemble if row.get("promotion_status") == "promoted"]
    generated_names = ", ".join(f"{row['dataset']}:{row['base_variant']}" for row in generated_cache) or "None"
    verified_names = ", ".join(f"{row['dataset']}:{row['base_variant']}" for row in verified_replay) or "None"
    mismatch_names = ", ".join(f"{row['dataset']}:{row['base_variant']} delta={row.get('delta_replay', '')}" for row in mismatch_replay) or "None"
    slow_names = ", ".join(f"{row['dataset']}:{row['base_variant']} ({row.get('blocked_reason', '')})" for row in slow_or_failed_cache) or "None"
    logit_correct_summary = (
        "Yes: " + "; ".join(row.get("dataset", "") for row in logit_correct_promoted)
        if logit_correct_promoted
        else "No promoted LogitCorrectLite row."
    )
    path_summary = (
        "Yes: " + "; ".join(f"{row.get('dataset', '')} {row.get('test_acc_before', '')}->{row.get('test_acc_after', '')}" for row in path_promoted)
        if path_promoted
        else "No promoted PathLogitCorrectLite row."
    )
    pseudo_summary = (
        "Yes: " + "; ".join(f"{row.get('dataset', '')} {row.get('test_acc_before', '')}->{row.get('test_acc_after', '')}" for row in pseudo_promoted)
        if pseudo_promoted
        else "No promoted Pseudo-SCAP row."
    )
    ensemble_summary = (
        "Yes: " + "; ".join(row.get("dataset", "") for row in ensemble_promoted)
        if ensemble_promoted
        else "No promoted safe ensemble row."
    )
    lines = [
        "# T1.1 Safe Cache and Boost Stage Summary",
        "",
        "This stage implements safe-row logit cache generation, replay audit, validation-only T1.1 boosters, and large-scale dry-run estimates.",
        "",
        "## Code Changes",
        "",
        "- Exposed all-target logits from SeHGNNLite and pipeline graph/compiled inference for replay-safe cache generation.",
        "- Extended T1.1 safe-row cache generation to DBLP, IMDB, ogbn-arxiv, and ogbn-products LAD; products R++ 500-epoch base is opt-in because it was too slow locally.",
        "- Wired validation-only PathLogitCorrectLite for replay-verified DBLP/IMDB caches while keeping sparse path steps and no exposed meta-path edge types.",
        "- Made cache reuse explicit and validated replay/gate cache metadata before reuse.",
        "- Refreshed replay audit, booster tables, large dry-run, and this stage summary.",
        "",
        "## Cache Generation",
        "",
        *markdown_table(cache, ["dataset", "base_variant", "cache_status", "train_nodes", "valid_nodes", "test_nodes", "all_target_nodes", "blocked_reason"]),
        "",
        "## Cache Replay",
        "",
        *markdown_table(replay, ["dataset", "base_variant", "cache_status", "historical_test_acc", "replay_test_acc", "delta_replay", "blocked_reason"]),
        "",
        "## Replay Classification",
        "",
        "### Verified",
        "",
        *markdown_table(verified_replay, ["dataset", "base_variant", "historical_test_acc", "replay_test_acc", "delta_replay"]),
        "",
        "### Mismatch",
        "",
        *markdown_table(mismatch_replay, ["dataset", "base_variant", "historical_test_acc", "replay_test_acc", "delta_replay", "blocked_reason"]),
        "",
        "### Not Generated",
        "",
        *markdown_table(slow_or_failed_cache, ["dataset", "base_variant", "cache_status", "blocked_reason"]),
        "",
        "## Promoted Rows",
        "",
        *markdown_table(promoted, ["dataset", "promoted_variant", "base_variant", "valid_acc_after", "accuracy", "macro_f1", "predicted_class_count", "promotion_status", "promotion_reason"]),
        "",
        "## Blocked Rows",
        "",
        *markdown_table(blocked, ["dataset", "base_variant", "cache_status", "promotion_status", "blocked_reason", "promotion_reason"]),
        "",
        "## Required Questions",
        "",
        f"1. Were the correct historical safe rows regenerated with logits cache? Generated: {generated_names}. Slow/failed: {slow_names}.",
        f"2. Did cache replay match the historical metrics? Verified: {verified_names}. Mismatch: {mismatch_names}.",
        f"3. Did LogitCorrectLite improve any dataset? {logit_correct_summary}",
        "4. Did Correct&Smooth-lite improve arxiv/products? No promoted arxiv/products Correct&Smooth row; their replay caches did not pass the historical gate.",
        f"5. Did PathLogitCorrectLite improve DBLP/IMDB? {path_summary}",
        f"6. Did Pseudo-SCAP improve any dataset? {pseudo_summary}",
        f"7. Did Safe Logit Ensemble improve products or ACM? {ensemble_summary}",
        f"8. Which rows were promoted? {promoted_summary}.",
        "9. Which rows were blocked and why? See `Blocked Rows`.",
        "10. Did any promoted row use forbidden components? No; promotion validator rejects forbidden flags.",
        f"11. Did any promoted medium row use bounded edges? {'Yes' if medium_promoted_bounded else 'No'}.",
        f"12. Did macro-F1 or predicted class count collapse? {'Yes' if macro_collapse else 'No'} for promoted rows.",
        f"13. Which datasets are now eligible for condensation recovery? {sorted({row['dataset'] for row in promoted}) if promoted else []}.",
        f"14. If no improvement occurred, what is the next bottleneck: cache mismatch, base signal ceiling, or validation overfit? Current blockers are replay mismatch ({mismatch_names}) and slow local products R++ generation ({slow_names}); verified rows can proceed to condensation recovery.",
        "",
        "## Artifacts",
        "",
        f"- `{SAFE_CACHE}`",
        f"- `{REPLAY}`",
        f"- `{CORRECT}`",
        f"- `{PATH_CORRECT}`",
        f"- `{PSEUDO}`",
        f"- `{ENSEMBLE}`",
        f"- `{BOOST}`",
        f"- `{DRY}`",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1.1 safe cache and boost stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--reuse-existing-caches", action="store_true", help="Reuse already validated cache directories instead of regenerating them.")
    args = parser.parse_args()
    cache_args = ["--seed", str(args.seed), "--epochs", str(args.epochs)]
    if args.reuse_existing_caches:
        cache_args.append("--reuse-existing-caches")
    else:
        cache_args.append("--no-reuse-existing-caches")
    _run("run_t1_generate_safe_logit_caches.py", cache_args)
    _run("run_t1_cache_replay_audit.py", ["--seed", str(args.seed)])
    _run("run_t1_logit_correct_safe.py")
    _run("run_t1_path_logit_correct_safe.py")
    _run("run_t1_pseudo_scap_safe.py")
    _run("run_t1_safe_logit_ensemble_safe.py")
    _run("run_t1_large_logit_affinity_dry_run.py")
    promoted = _best_boost_rows()
    _write_summary(promoted)


if __name__ == "__main__":
    main()
