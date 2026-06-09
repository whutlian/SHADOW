from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: str | Path) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best(rows: list[dict], dataset: str) -> dict | None:
    candidates = []
    for row in rows:
        if row.get("dataset") != dataset or row.get("status", "completed") != "completed":
            continue
        acc = _float(row.get("accuracy"))
        if acc is not None:
            candidates.append((acc, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _table(rows: list[dict], fields: list[str]) -> list[str]:
    if not rows:
        return ["None."]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SOTA alignment clean sprint summary.")
    parser.add_argument("--pytest-summary", default="")
    parser.add_argument("--output", default="experiments/reports/sota_alignment_clean_sprint_summary.md")
    args = parser.parse_args()

    audit = _read_csv("experiments/tables/sota_audit_seed42.csv")
    fullgraph = _read_csv("experiments/tables/fullgraph_backbone_audit_seed42.csv")
    dblp_schema = _read_csv("experiments/tables/dblp_schema_audit_seed42.csv")
    clean = _read_csv("experiments/tables/sota_clean_small_seed42.csv")
    medium = _read_csv("experiments/tables/medium_no_diffusion_refine_seed42.csv")
    teacher = _read_csv("experiments/tables/teacher_herding_kd_gated_seed42.csv")

    audit_counts: dict[str, int] = {}
    for row in audit:
        audit_counts[row.get("status", "")] = audit_counts.get(row.get("status", ""), 0) + 1
    audit_count_rows = [{"status": key, "count": value} for key, value in sorted(audit_counts.items())]

    clean_best = [row for dataset in ["acm", "dblp", "imdb"] if (row := _best(clean, dataset)) is not None]
    medium_best = [row for dataset in ["ogbn-arxiv", "ogbn-products"] if (row := _best(medium, dataset)) is not None]
    invalid_examples = [row for row in audit if row.get("status") != "completed"][:20]

    lines = [
        "# SOTA Alignment Clean Sprint Summary",
        "",
        "## Scope",
        "",
        "- Seed policy: single seed `42` only.",
        "- Default method remains frozen as `Shadow-HGC-R-1`; all SOTA alignment paths are explicit scripts/diagnostics.",
        "- Diffusion, CoverageMedoid, source anchors, and old KD are not promoted.",
        "- Invalid historical rows are retained in audit artifacts but excluded from best-row summaries.",
        "",
        "## Code Changes",
        "",
        "- Added hard audit gates in `shadow_hgc/audit/*` and read-only `scripts/run_sota_audit.py`.",
        "- Added schema-default meta-path specs and DBLP schema audit.",
        "- Added Path-LAD v2 diagnostics, `P2` target-target Path-LAD support, two-hop LAD utility, teacher-demand herding selector, and KD v2 gate/loss.",
        "- Added actual SeHGNNLite target-row/prototype training utilities for clean small/fullgraph audits.",
        "- Added clean experiment scripts for fullgraph backbone, clean small, medium no-diffusion refine, gated teacher/KD diagnostics, and this summary.",
        "- Updated small dataset YAML configs to match the loader-exposed relations.",
        "",
        "## Pytest",
        "",
        args.pytest_summary or "Pytest summary is recorded after the final verification run.",
        "",
        "## Hard Audit Status",
        "",
    ]
    lines.extend(_table(audit_count_rows, ["status", "count"]))
    lines.extend([
        "",
        "## Fullgraph Backbone Audit",
        "",
    ])
    lines.extend(_table(fullgraph, ["dataset", "variant", "accuracy", "macro_f1", "target_gate", "gate_passed", "blocked_by_fullgraph_backbone", "status"]))
    lines.extend([
        "",
        "Interpretation: ACM and arxiv passed the first backbone gates. DBLP, IMDB, and products are marked as backbone/data constrained for SOTA chasing in this sprint.",
        "",
        "## DBLP Schema Audit",
        "",
    ])
    lines.extend(_table(dblp_schema, ["target_type", "label_node_type", "apa_available", "computed_metapath_blocks", "skipped_metapath_blocks", "hard_requirements_passed", "notes"]))
    lines.extend([
        "",
        "## Clean Small Results",
        "",
        "Best completed clean row per small dataset:",
    ])
    lines.extend(_table(clean_best, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "predicted_class_count", "total_condensed_node_ratio", "model_type", "status"]))
    lines.extend([
        "",
        "All clean small rows:",
    ])
    lines.extend(_table(clean, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "predicted_class_count", "total_condensed_node_ratio", "model_type", "status"]))
    lines.extend([
        "",
        "## Medium No-Diffusion Refine",
        "",
        "Best completed medium row per dataset:",
    ])
    lines.extend(_table(medium_best, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "predicted_class_count", "total_condensed_node_ratio", "status"]))
    lines.extend([
        "",
        "All medium rows:",
    ])
    lines.extend(_table(medium, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "predicted_class_count", "two_hop_lad_blocks", "status", "reason"]))
    lines.extend([
        "",
        "## Teacher Herding / KD Gates",
        "",
    ])
    lines.extend(_table(teacher, ["dataset", "variant", "status", "kd_gate_passed", "kd_skip_reason", "teacher_type"]))
    lines.extend([
        "",
        "## Invalid Examples",
        "",
    ])
    lines.extend(_table(invalid_examples, ["dataset", "variant", "status", "invalid_reasons", "source_log"]))
    lines.extend([
        "",
        "## Conclusions",
        "",
        "- Promote: audit gates, schema-default clean SeHGNNLite for ACM, DBLP APA schema audit, and no-diffusion LAD reference rows.",
        "- Keep as diagnostic: Path-LAD v2 feature blocks on IMDB; they are train-label-only and valid but did not beat clean MAM/MDM/MKM.",
        "- Drop from promoted path: CoverageMedoid, old KD, source anchors, products P2 LAD as currently implemented, and high-dimensional diffusion.",
        "- Bottlenecks: DBLP/IMDB fullgraph backbone capacity is below requested gates; arxiv two-hop LAD caused large regression and class collapse; products two-hop LAD hit CPU OOM due a 23GB allocation path and one row timed out.",
        "",
        "## Artifact Files",
        "",
        "- `experiments/tables/sota_audit_seed42.csv`",
        "- `experiments/tables/fullgraph_backbone_audit_seed42.csv`",
        "- `experiments/tables/dblp_schema_audit_seed42.csv`",
        "- `experiments/tables/sota_clean_small_seed42.csv`",
        "- `experiments/tables/medium_no_diffusion_refine_seed42.csv`",
        "- `experiments/tables/teacher_herding_kd_gated_seed42.csv`",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

