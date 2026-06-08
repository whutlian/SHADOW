from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _gate(status: str, title: str, reason: str) -> str:
    return f"| {title} | {status} | {reason} |"


def _toy_gate(toy_dir: Path) -> tuple[str, str]:
    required = [
        "summary_refined.json",
        "summary_refined_private_shadow.json",
        "summary_refined_self_only.json",
        "summary_refined_full_graph.json",
    ]
    payloads = [_json(toy_dir / name) for name in required]
    if any(payload is None for payload in payloads):
        return "WARN", "Some canonical toy logs are missing."
    ok = all(_float(payload.get("accuracy"), 0.0) == 1.0 and _float(payload.get("macro_f1"), 0.0) == 1.0 for payload in payloads if payload)
    return ("PASS", "Toy main/private/self/full all have 1.0 accuracy and macro-F1.") if ok else ("FAIL", "At least one toy mode is below 1.0.")


def _small_gate(rows: list[dict]) -> tuple[str, str]:
    if not rows:
        return "WARN", "No small ratio rows found."
    datasets = sorted({row["dataset"] for row in rows if row.get("dataset")})
    wins = 0
    self_gap = []
    imdb_fail = False
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset and row.get("status", "completed") == "completed"]
        shadow = [row for row in subset if row.get("method") == "Shadow-HGC-R-1"]
        classical = [
            row for row in subset
            if row.get("method") in {"Random-HG", "Herding-HG", "K-Center-HG"}
            and row.get("baseline_match_mode", "") in {"", "target_ratio"}
        ]
        self_only = [row for row in subset if row.get("method") == "Self-Only-MLP"]
        best_shadow = max((_float(row.get("accuracy"), -1.0) for row in shadow), default=-1.0)
        best_classical = max((_float(row.get("accuracy"), -1.0) for row in classical), default=-1.0)
        best_self = max((_float(row.get("accuracy"), -1.0) for row in self_only), default=-1.0)
        if best_shadow >= best_classical and best_classical >= 0:
            wins += 1
        if best_self - best_shadow > 0.03:
            self_gap.append(dataset)
        if dataset == "imdb" and best_shadow < best_self and best_shadow < best_classical:
            imdb_fail = True
    if imdb_fail:
        return "FAIL", "IMDB remains below both self-only and best classical baseline."
    if wins >= 2:
        status = "WARN" if self_gap else "PASS"
        detail = f"Shadow matches/beats best classical baseline on {wins}/{len(datasets)} datasets."
        if self_gap:
            detail += f" Self-only gap >3 points on: {', '.join(self_gap)}."
        return status, detail
    return "WARN", f"Shadow matches/beats best classical baseline on only {wins}/{len(datasets)} datasets."


def _medium_gate(rows: list[dict]) -> tuple[str, str]:
    if not rows:
        return "WARN", "No medium ratio rows found."
    products = [
        row for row in rows
        if row.get("dataset") == "ogbn-products" and row.get("method") == "Shadow-HGC-R-1" and row.get("status", "completed") == "completed"
    ]
    arxiv = [
        row for row in rows
        if row.get("dataset") == "ogbn-arxiv" and row.get("method") == "Shadow-HGC-R-1" and row.get("status", "completed") == "completed"
    ]
    if products:
        products_sorted = sorted(products, key=lambda row: _float(row.get("ratio"), _float(row.get("requested_target_budget"), 0.0)))
        product_acc = [_float(row.get("accuracy"), 0.0) for row in products_sorted]
        near_mono = all(product_acc[i] + 0.02 >= product_acc[i - 1] for i in range(1, len(product_acc)))
    else:
        near_mono = False
    class_collapse = any(int(float(row.get("predicted_classes", 0) or 0)) <= 1 for row in products + arxiv)
    if class_collapse:
        return "FAIL", "At least one medium run collapsed to one predicted class."
    if near_mono:
        return "WARN", "Products is monotonic/near-monotonic; arxiv still needs comparison against self/private/full graph."
    return "WARN", "Medium ratio evidence is incomplete or products is not monotonic."


def _dry_run_gate(paths: list[Path]) -> tuple[str, str]:
    payloads = [_json(path) for path in paths if path.exists()]
    payloads = [payload for payload in payloads if payload]
    if not payloads:
        return "WARN", "No ratio-aware dry-run logs found."
    for payload in payloads:
        estimates = payload.get("ratio_estimates", [])
        if not estimates:
            return "FAIL", f"{payload.get('dataset', 'unknown')} has no ratio_estimates."
        required = {
            "demand_cache_GB",
            "edge_slice_cache_GB",
            "peak_ram_estimate_GB",
            "disk_bytes_estimate_GB",
            "full_edge_scans",
            "cache_all_targets",
        }
        for row in estimates:
            missing = required - set(row)
            if missing:
                return "FAIL", f"Dry-run estimate is missing {sorted(missing)}."
            if row.get("cache_all_targets") is not False:
                return "FAIL", "Dry-run reports cache_all_targets != false."
    return "PASS", "Ratio-aware dry-run logs contain memory/disk/scan fields and cache_all_targets=false."


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 0-4 solidness report.")
    parser.add_argument("--small-ratio-table", default="experiments/tables/small_ratio_main.csv")
    parser.add_argument("--medium-ratio-table", default="experiments/tables/medium_ratio_main.csv")
    parser.add_argument("--toy-dir", default="experiments/logs/toy")
    parser.add_argument("--dry-run-logs", nargs="*", default=[
        "experiments/logs/scaling_stress/dry_run_papers100m_ratio.json",
        "experiments/logs/scaling_stress/dry_run_mag240m_ratio.json",
        "experiments/logs/scaling_stress/dry_run_ratio_smoke.json",
    ])
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--output", default="experiments/reports/stage0_4_solidness_report.md")
    args = parser.parse_args()

    small_rows = _rows(args.small_ratio_table)
    medium_rows = _rows(args.medium_ratio_table)
    gate0 = ("PASS", "Full current test suite passed in this run.") if args.tests_passed else ("WARN", "Report was generated without --tests-passed.")
    gate1 = _toy_gate(Path(args.toy_dir))
    gate2 = _small_gate(small_rows)
    gate3 = _medium_gate(medium_rows)
    gate4 = _dry_run_gate([Path(path) for path in args.dry_run_logs])

    stage5 = "UNBLOCKED" if gate0[0] == "PASS" and gate1[0] == "PASS" and gate4[0] == "PASS" and gate2[0] != "FAIL" and gate3[0] != "FAIL" else "BLOCKED"
    lines = [
        "# Stage 0-4 Solidness Report",
        "",
        f"Stage 5 status: **{stage5}**",
        "",
        "| Gate | Status | Reason |",
        "| --- | --- | --- |",
        _gate(gate0[0], "Gate 0: tests", gate0[1]),
        _gate(gate1[0], "Gate 1: toy", gate1[1]),
        _gate(gate2[0], "Gate 2: small datasets", gate2[1]),
        _gate(gate3[0], "Gate 3: medium datasets", gate3[1]),
        _gate(gate4[0], "Gate 4: I/O dry run", gate4[1]),
        "",
        "## Required Interpretation",
        "",
        f"1. Target-ratio matched baseline status: {gate2[1]}",
        "2. Total-node matched comparison is available when `baseline_match_mode=total_condensed_nodes` appears in `small_ratio_main.csv`.",
        "3. IMDB is treated as a visible failure case if it remains below self-only and K-Center.",
        "4. relation_linear vs relation_mlp should be read from rows with `model=relation_linear` and `model=relation_mlp`.",
        f"5. Products ratio scaling: {gate3[1]}",
        "6. Arxiv remains a diagnostic target when below self-only/private/full graph by large margins.",
        f"7. Ratio dry-run status: {gate4[1]}",
        "8. Ratio logs use `r...` names; count-mode compatibility uses `count...` names.",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    print(f"stage5_status={stage5}")


if __name__ == "__main__":
    main()
