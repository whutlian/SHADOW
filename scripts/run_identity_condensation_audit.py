from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.eval.gap_decomposition import decompose_condensation_gaps


FIELDS = [
    "dataset",
    "ratio",
    "seed",
    "status",
    "fullgraph_acc",
    "identity_condensed_acc",
    "prototype_oracle_acc",
    "shadow_hgc_acc",
    "full_to_identity_gap",
    "identity_to_oracle_gap",
    "oracle_to_shadow_gap",
    "full_to_shadow_gap",
    "bottleneck_label",
    "fullgraph_variant",
    "identity_variant",
    "prototype_oracle_variant",
    "shadow_variant",
    "fullgraph_source_log",
    "identity_source_log",
    "prototype_oracle_source_log",
    "shadow_source_log",
    "schema_compatible",
    "compatibility_reason",
    "condensed_path_inconsistent",
    "reason",
]


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


def _best(rows: list[dict], dataset: str, *, variant_contains: str | None = None, ratio: float | None = None) -> dict | None:
    candidates = []
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        if row.get("status", "completed") not in {"completed", "completed_existing_diagnostic"}:
            continue
        if variant_contains and variant_contains not in row.get("variant", ""):
            continue
        if ratio is not None:
            row_ratio = _float(row.get("ratio", row.get("requested_ratio")))
            if row_ratio is None or abs(row_ratio - float(ratio)) > 1e-9:
                continue
        acc = _float(row.get("accuracy"))
        if acc is not None:
            candidates.append((acc, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _source_log(row: dict | None) -> str:
    if row is None:
        return ""
    return row.get("source_log", row.get("log_path", ""))


def _compatibility(dataset: str, identity: dict | None, oracle: dict | None, shadow: dict | None) -> tuple[bool, str]:
    if identity is None or oracle is None or shadow is None:
        return False, "missing identity/oracle/shadow row"
    if dataset == "imdb":
        return False, "existing IMDB identity/oracle diagnostics use label_affinity while the best shadow row uses label_affinity_metapath"
    return True, "compatible_existing_rows"


def _row(dataset: str, seed: int, fullgraph: list[dict], clean: list[dict], medium: list[dict], diag: list[dict], *, ratio: float | None = None) -> dict:
    fg = _best(fullgraph, dataset)
    fullgraph_acc = _float(fg.get("accuracy")) if fg else None
    fg_passed = bool(str(fg.get("gate_passed", "")).lower() == "true") if fg else False
    if dataset in {"acm", "dblp", "imdb"}:
        shadow = _best(clean, dataset, ratio=ratio) if ratio is not None else _best(clean, dataset)
    else:
        shadow = _best(medium, dataset, variant_contains="LAD_reference", ratio=ratio) if ratio is not None else _best(medium, dataset, variant_contains="LAD_reference")
    oracle = _best(diag, dataset, variant_contains="PrototypeOracle", ratio=ratio) if ratio is not None else _best(diag, dataset, variant_contains="PrototypeOracle")
    full_demand = _best(diag, dataset, variant_contains="FullDemandTable", ratio=ratio) if ratio is not None else _best(diag, dataset, variant_contains="FullDemandTable")
    identity_acc = _float(full_demand.get("accuracy")) if full_demand else fullgraph_acc
    prototype_oracle_acc = _float(oracle.get("accuracy")) if oracle else None
    shadow_acc = _float(shadow.get("accuracy")) if shadow else None
    schema_compatible, compatibility_reason = _compatibility(dataset, full_demand or fg, oracle, shadow)
    gaps = decompose_condensation_gaps(
        fullgraph_acc=fullgraph_acc,
        identity_condensed_acc=identity_acc,
        prototype_oracle_acc=prototype_oracle_acc,
        shadow_hgc_acc=shadow_acc,
        fullgraph_gate_passed=fg_passed,
    )
    return {
        "dataset": dataset,
        "ratio": "" if ratio is None else ratio,
        "seed": seed,
        "status": "completed" if fg and shadow else "missing_inputs",
        **gaps,
        "fullgraph_variant": "" if fg is None else fg.get("variant", ""),
        "identity_variant": "FullDemandTable-MLP" if full_demand else "fullgraph_backbone_reference_proxy",
        "prototype_oracle_variant": "" if oracle is None else oracle.get("variant", ""),
        "shadow_variant": "" if shadow is None else shadow.get("variant", ""),
        "fullgraph_source_log": _source_log(fg),
        "identity_source_log": _source_log(full_demand),
        "prototype_oracle_source_log": _source_log(oracle),
        "shadow_source_log": _source_log(shadow),
        "schema_compatible": schema_compatible,
        "compatibility_reason": compatibility_reason,
        "condensed_path_inconsistent": gaps["bottleneck_label"] == "condensed_path_inconsistent",
        "reason": "" if fg and shadow else "missing fullgraph or shadow summary row",
    }


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = [
        "# Identity Condensation Audit Seed 42",
        "",
        "| Dataset | Ratio | Fullgraph | Identity | Oracle | Shadow | Full->Shadow | Compatible | Bottleneck | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row.get('ratio','')} | {row.get('fullgraph_acc','')} | {row.get('identity_condensed_acc','')} | "
            f"{row.get('prototype_oracle_acc','')} | {row.get('shadow_hgc_acc','')} | {row.get('full_to_shadow_gap','')} | "
            f"{row.get('schema_compatible','')} | {row.get('bottleneck_label','')} | {row.get('status','')} |"
        )
    lines.extend([
        "",
        "Identity rows use exact/full-demand diagnostics when available; otherwise they are explicit proxies or missing-input rows.",
        "Rows with schema/config mismatch are retained for diagnosis but excluded from promoted best-row conclusions.",
        "",
        f"- CSV: `{csv_path}`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build identity/oracle/shadow condensation gap decomposition.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/tables/identity_condensation_audit_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/identity_condensation_audit_seed42.md")
    args = parser.parse_args()
    fullgraph = _read_csv("experiments/tables/fullgraph_parity_seed42.csv")
    clean = _read_csv("experiments/tables/sota_clean_small_seed42.csv")
    medium = _read_csv("experiments/tables/medium_no_diffusion_refine_seed42.csv")
    diag = _read_csv("experiments/tables/lad_stage_diagnostics_seed42.csv")
    rows = [
        _row("acm", args.seed, fullgraph, clean, medium, diag, ratio=0.12),
        _row("dblp", args.seed, fullgraph, clean, medium, diag, ratio=0.096),
        _row("imdb", args.seed, fullgraph, clean, medium, diag, ratio=0.05),
        _row("ogbn-arxiv", args.seed, fullgraph, clean, medium, diag, ratio=0.12),
        _row("ogbn-products", args.seed, fullgraph, clean, medium, diag, ratio=0.12),
    ]
    output = Path(args.output)
    write_csv(output, rows, FIELDS)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
