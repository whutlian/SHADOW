from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SMALL_CSV = Path("experiments/tables/lad_stage_small_seed42.csv")
MEDIUM_CSV = Path("experiments/tables/lad_stage_medium_seed42.csv")
DIAG_CSV = Path("experiments/tables/lad_stage_diagnostics_seed42.csv")
REPORT = Path("experiments/reports/lad_stage_summary.md")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str, default: float = -1.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _fmt(value: object) -> str:
    if value in {"", None}:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _best(rows: list[dict], dataset: str) -> dict | None:
    subset = [row for row in rows if row.get("dataset") == dataset and row.get("status") == "completed" and row.get("accuracy") not in {"", None}]
    if not subset:
        return None
    return max(subset, key=lambda row: _float(row, "accuracy"))


def _table(rows: list[dict], *, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Dataset | Ratio | Variant | Status | Acc | Macro-F1 | Pred classes | Compiled | LAD | Boundary |",
        "|---|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset','')} | {_fmt(row.get('ratio'))} | {row.get('variant','')} | {row.get('status','')} | "
            f"{_fmt(row.get('accuracy'))} | {_fmt(row.get('macro_f1'))} | {row.get('predicted_class_count','')} | "
            f"{row.get('compiled_head','')} | {row.get('label_affinity','')} | {row.get('boundary_prototypes','')} |"
        )
    return lines


def build_report() -> None:
    small = _read_rows(SMALL_CSV)
    medium = _read_rows(MEDIUM_CSV)
    diagnostics = _read_rows(DIAG_CSV)
    all_rows = small + medium
    best_lines = []
    for dataset in ["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"]:
        best = _best(all_rows, dataset)
        if best is None:
            best_lines.append(f"- {dataset}: no completed row.")
        else:
            best_lines.append(
                f"- {dataset}: `{best.get('variant')}` at ratio `{_fmt(best.get('ratio'))}` reached "
                f"accuracy `{_fmt(best.get('accuracy'))}` / macro-F1 `{_fmt(best.get('macro_f1'))}`."
            )
    diag_completed = [row for row in diagnostics if row.get("status") == "completed"]
    lines = [
        "# Shadow-HGC-L / LAD Stage Summary",
        "",
        "## 1. Scope",
        "",
        "- Seed policy: single seed `42` only.",
        "- No-diffusion decision: diffusion is not promoted because it caused OOM/resource failures on products and is too expensive for large-scale goals. It remains an appendix diagnostic only.",
        "- Variants: V0 current_best, V1 compiled demand head, V2 compiled demand head + LAD, V3 V2 + boundary-aware prototypes.",
        "- Ratios: ACM 9.6%; DBLP 0.5% and 6.5%; IMDB 0.5%, 2.5%, 5.0%; ogbn-arxiv/products 6.0% and 12.0%.",
        "",
        "## 2. Code Changes",
        "",
        "- Added train-label-only LAD feature computation in `shadow_hgc/features/label_affinity.py`.",
        "- Added compiled demand schema/table helpers in `shadow_hgc/features/compiled_table.py`.",
        "- Added block-gated compiled demand MLP in `shadow_hgc/models/compiled_demand.py`.",
        "- Added boundary-aware prototype helper in `shadow_hgc/prototype/boundary.py`.",
        "- Integrated opt-in LAD/compiled/boundary arguments into `shadow_hgc/pipeline/core.py` without changing the default R-1 path.",
        "- Added LAD scripts under `scripts/run_lad_*.py` and tests under `tests/test_*lad*`, `tests/test_compiled_*`, and `tests/test_boundary_*`.",
        "- Test command: `C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe -m pytest tests -q`.",
        "",
        "## 3. Main Results",
        "",
        *best_lines,
        "",
        *_table(small, title="Small Datasets"),
        "",
        *_table(medium, title="Medium Datasets"),
        "",
        "## 4. Diagnostic Results",
        "",
        *_table(diagnostics, title="Upper-Bound Diagnostics"),
        "",
        "Interpretation template:",
        "",
        "- If FullDemandTable is high but condensed LAD is low, target prototype condensation is the bottleneck.",
        "- If PrototypeOracleDemand is high but shadow reconstructed compiled demand is low, shadow factorization is the bottleneck.",
        "- If both diagnostics are low, the bottleneck is insufficient demand signal or training head capacity.",
        "",
        f"Completed diagnostic rows: {len(diag_completed)} / {len(diagnostics)}.",
        "",
        "## 5. LAD Analysis",
        "",
        "- LAD uses training labels only; validation/test labels are not used in label-affinity construction.",
        "- LAD blocks are target-side compiled features, not exposed graph edge types.",
        "- LAD block statistics and learned block gates are logged in per-run JSON files.",
        "",
        "## 6. Boundary Prototype Analysis",
        "",
        "- V3 enables boundary-aware prototypes with `boundary_fraction=0.30` and train-only entropy scoring.",
        "- Boundary pool sizes, score stats, and base/boundary prototype counts are logged in V3 JSON files.",
        "",
        "## 7. Compression and Resource Accounting",
        "",
        "- Tables include target ratio, total condensed node ratio, byte-size compression, LAD precompute time, CPU RAM, and GPU RAM fields.",
        "- FullDemandTable diagnostics are upper bounds and should not be read as condensation compression results.",
        "",
        "## 8. Decision",
        "",
        "- Promote / do not promote LAD: decide from V2 versus V1 and diagnostics above.",
        "- Promote / do not promote compiled head: decide from V1 versus V0.",
        "- Promote / do not promote boundary prototypes: decide from V3 versus V2.",
        "- Return to large-scale stage only if no-diffusion LAD/compiled rows meet medium gates or diagnostics identify a clear fix.",
        "",
        "Direct bottleneck answers:",
        "",
        "- Is the bottleneck condensation? See FullDemandTable versus V2/V3 rows.",
        "- Is the bottleneck shadow factorization? See PrototypeOracleDemand versus V2/V3 rows.",
        "- Is the bottleneck training head? See V1 and FullDemandTable rows.",
        "- Is LAD useful enough to replace diffusion? See V2/V3 no-diffusion medium rows.",
        "",
        "## 9. Next Recommended Experiments",
        "",
        "- Run multi-seed only for rows that beat R++ without diffusion.",
        "- If products remains below target, avoid diffusion and focus on sparse train-label affinity plus target coreset allocation.",
        "- If PrototypeOracleDemand is much better than V2/V3, improve shadow reconstruction before adding model capacity.",
        "",
        "## Files",
        "",
        f"- Small CSV: `{SMALL_CSV}`",
        f"- Medium CSV: `{MEDIUM_CSV}`",
        f"- Diagnostics CSV: `{DIAG_CSV}`",
        f"- Report: `{REPORT}`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full LAD stage and build report.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-report", action="store_true")
    args = parser.parse_args()
    if not args.only_report:
        base = [sys.executable]
        common = ["--seed", str(args.seed), "--epochs", str(args.epochs)]
        if args.skip_existing:
            common.append("--skip-existing")
        _run(base + ["scripts/run_lad_small.py", *common])
        medium = base + ["scripts/run_lad_medium.py", *common]
        diagnostics = base + ["scripts/run_lad_diagnostics.py", *common]
        if args.download:
            medium.append("--download")
            diagnostics.append("--download")
        _run(medium)
        _run(diagnostics)
    build_report()


if __name__ == "__main__":
    main()
