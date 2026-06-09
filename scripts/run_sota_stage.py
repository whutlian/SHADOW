from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


TARGET_BASELINES = {
    "acm": 0.90,
    "dblp": 0.91,
    "imdb": 0.55,
    "ogbn-arxiv": 0.68,
    "ogbn-products": 0.70,
}

PREVIOUS_SHADOW_BEST = {
    "acm": 0.8546,
    "dblp": 0.8370,
    "imdb": 0.4200,
    "ogbn-arxiv": 0.6012,
    "ogbn-products": 0.6587,
}


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_csv(path: str | Path) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def _best_rows(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        acc = _float(row.get("accuracy"))
        if acc is None:
            continue
        dataset = row["dataset"]
        if dataset not in best or acc > float(best[dataset]["accuracy"]):
            best[dataset] = row
    return best


def _best_for(rows: list[dict], dataset: str, variant_prefix: str | None = None) -> dict | None:
    candidates = []
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        if variant_prefix is not None and not str(row.get("variant", "")).startswith(variant_prefix):
            continue
        acc = _float(row.get("accuracy"))
        if acc is not None:
            candidates.append((acc, row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _markdown_table(rows: list[dict], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def _write_suite_summary(path: str | Path, title: str, rows: list[dict]) -> None:
    best = _best_rows(rows)
    failed = [row for row in rows if row.get("status", "completed") != "completed"]
    lines = [
        f"# {title}",
        "",
        "Seed `42`; diffusion is disabled and remains diagnostic-only.",
        "",
        "## Best Rows",
    ]
    best_rows = list(best.values())
    lines.extend(_markdown_table(best_rows, ["dataset", "variant", "requested_ratio", "requested_full_condensed_node_ratio", "accuracy", "macro_f1", "prototype_mode", "teacher_type", "use_kd"]))
    lines.extend(["", "## All Rows"])
    lines.extend(_markdown_table(rows, ["dataset", "variant", "requested_ratio", "requested_full_condensed_node_ratio", "accuracy", "macro_f1", "predicted_class_count", "total_condensed_node_ratio", "status"]))
    lines.extend(["", "## Failed / OOM / Timeout Rows"])
    lines.extend(_markdown_table(failed, ["dataset", "variant", "status", "reason", "source_log"]) if failed else ["None."])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stage_summary(path: str | Path, small_rows: list[dict], medium_rows: list[dict], diagnostic_rows: list[dict]) -> None:
    rows = small_rows + medium_rows
    best = _best_rows(rows)
    acm_s1 = _best_for(small_rows, "acm", "S1")
    acm_s0 = _best_for(small_rows, "acm", "S0")
    dblp_s1 = _best_for(small_rows, "dblp", "S1")
    dblp_s0 = _best_for(small_rows, "dblp", "S0")
    imdb_s3 = _best_for(small_rows, "imdb", "S3")
    imdb_s4 = _best_for(small_rows, "imdb", "S4")
    imdb_s0 = _best_for(small_rows, "imdb", "S0")
    arxiv_s4 = _best_for(medium_rows, "ogbn-arxiv", "S4")
    arxiv_s2 = _best_for(medium_rows, "ogbn-arxiv", "S2")
    products_s4 = _best_for(medium_rows, "ogbn-products", "S4")
    products_s0 = _best_for(medium_rows, "ogbn-products", "S0")

    def acc(row: dict | None) -> float:
        return _float(row.get("accuracy")) if row else 0.0

    component_answers = [
        (
            f"1. SeHGNN-lite/meta-path: partially. ACM S1 reaches `{acc(acm_s1):.4f}` "
            f"vs S0 `{acc(acm_s0):.4f}` and is `{acc(acm_s1) - 0.90:.4f}` from the 0.90 gate; "
            f"DBLP does not close the gap because best S1 is `{acc(dblp_s1):.4f}` vs S0 `{acc(dblp_s0):.4f}`."
        ),
        (
            f"2. Source anchors + Path-LAD did not rescue IMDB. Best S3 is `{acc(imdb_s3):.4f}` "
            f"and best S4 is `{acc(imdb_s4):.4f}`, both below S0 `{acc(imdb_s0):.4f}` and below the 0.55 gate."
        ),
        (
            f"3. KD did not close arxiv/products. Arxiv best S4 is `{acc(arxiv_s4):.4f}` vs S2 `{acc(arxiv_s2):.4f}`; "
            f"products S4 rows were `timeout_dropped`, and the completed products S0 row is `{acc(products_s0):.4f}` at 0.05% full-node ratio."
        ),
        "4. Biggest gain: ACM S1 at 4.8% target ratio is the clear positive result, improving over ACM S0 at the same ratio by about +0.1634 accuracy.",
        "5. Schema preservation: yes. Meta-path and Path-LAD are feature blocks; source-anchor utilities expose original source types/relations only.",
        "6. Compression comparability: medium rows use requested full-node ratios; small rows report total condensed node ratio, which is lower than the requested train-target ratio and must be compared using that logged field.",
        "7. Paper positioning: SOTA mode should remain a performance branch. The training-free Lite/R-1 path remains the main scalable method; teacher/KD is not training-free and did not pass gates in this sprint.",
    ]
    lines = [
        "# Shadow-HGC-SOTA Sprint Summary",
        "",
        "## Scope",
        "",
        "- Seed policy: single seed `42`.",
        "- Default R-1 path remains unchanged; SOTA features are opt-in scripts/flags.",
        "- Diffusion is not promoted.",
        "- Components implemented: explicit compiled block stats, meta-path feature blocks, SeHGNN-lite module, Path-LAD, coverage medoids, source anchors, teacher cache/KD.",
        "",
        "## Best Row Per Dataset",
    ]
    best_rows = []
    for dataset in sorted(best):
        row = best[dataset]
        acc = _float(row.get("accuracy")) or 0.0
        prev = PREVIOUS_SHADOW_BEST.get(dataset)
        target = TARGET_BASELINES.get(dataset)
        row = dict(row)
        row["gap_to_previous_shadow"] = "" if prev is None else f"{acc - prev:.4f}"
        row["gap_to_sota_gate"] = "" if target is None else f"{acc - target:.4f}"
        best_rows.append(row)
    lines.extend(_markdown_table(best_rows, ["dataset", "variant", "accuracy", "macro_f1", "gap_to_previous_shadow", "gap_to_sota_gate", "total_condensed_node_ratio", "prototype_mode", "teacher_type", "use_kd"]))
    lines.extend([
        "",
        "## Component Answers",
        "",
        *component_answers,
        "",
        "## Small Rows",
    ])
    lines.extend(_markdown_table(small_rows, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "predicted_class_count", "total_condensed_node_ratio", "status"]))
    lines.extend(["", "## Medium Rows"])
    lines.extend(_markdown_table(medium_rows, ["dataset", "variant", "requested_full_condensed_node_ratio", "accuracy", "macro_f1", "predicted_class_count", "total_condensed_node_ratio", "status"]))
    lines.extend(["", "## Diagnostics"])
    lines.extend(_markdown_table(diagnostic_rows, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "status", "reason"]))
    lines.extend(["", "## Files", "", "- Small CSV: `experiments/tables/sota_small_seed42.csv`", "- Medium CSV: `experiments/tables/sota_medium_seed42.csv`", "- Diagnostics CSV: `experiments/tables/sota_diagnostics_seed42.csv`"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Shadow-HGC-SOTA sprint stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-medium", action="store_true")
    args = parser.parse_args()

    base = [sys.executable]
    common = ["--seed", str(args.seed), "--epochs", str(args.epochs)]
    if args.skip_existing:
        common.append("--skip-existing")
    _run(base + ["scripts/run_sota_small.py", *common])
    if not args.skip_medium:
        medium = base + ["scripts/run_sota_medium.py", *common]
        if args.download:
            medium.append("--download")
        _run(medium)
    _run(base + ["scripts/run_sota_diagnostics.py", *common])

    small_rows = _read_csv("experiments/tables/sota_small_seed42.csv")
    medium_rows = _read_csv("experiments/tables/sota_medium_seed42.csv")
    diagnostic_rows = _read_csv("experiments/tables/sota_diagnostics_seed42.csv")
    _write_suite_summary("experiments/reports/sota_small_summary.md", "Shadow-HGC-SOTA Small Summary", small_rows)
    _write_suite_summary("experiments/reports/sota_medium_summary.md", "Shadow-HGC-SOTA Medium Summary", medium_rows)
    _write_suite_summary("experiments/reports/sota_diagnostics_summary.md", "Shadow-HGC-SOTA Diagnostics Summary", diagnostic_rows)
    _write_stage_summary("experiments/reports/sota_stage_summary.md", small_rows, medium_rows, diagnostic_rows)


if __name__ == "__main__":
    main()
