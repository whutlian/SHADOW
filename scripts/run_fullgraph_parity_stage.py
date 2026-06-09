from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.audit.parity import validate_promoted_row
from shadow_hgc.data.small import load_processed_small_dataset, load_processed_small_dataset_full_schema
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.train.sehgnn_lite_target import build_schema_default_blocks, train_prototype_sehgnn_lite


SCHEMA_CSV = Path("experiments/tables/schema_alignment_audit_seed42.csv")
FULLGRAPH_CSV = Path("experiments/tables/fullgraph_parity_seed42.csv")
IDENTITY_CSV = Path("experiments/tables/identity_condensation_audit_seed42.csv")
SUMMARY = Path("experiments/reports/fullgraph_parity_condensation_recovery_summary.md")

ACM_CSV = Path("experiments/tables/acm_s1_clean_tuned_seed42.csv")
DBLP_CSV = Path("experiments/tables/dblp_schema_fixed_candidate_seed42.csv")
IMDB_CSV = Path("experiments/tables/imdb_fullgraph_first_candidate_seed42.csv")
ARXIV_CSV = Path("experiments/tables/arxiv_no_diffusion_recovery_seed42.csv")
PRODUCTS_CSV = Path("experiments/tables/products_no_diffusion_recovery_seed42.csv")

SMALL_CLEAN_CSV = Path("experiments/tables/sota_clean_small_seed42.csv")
MEDIUM_CLEAN_CSV = Path("experiments/tables/medium_no_diffusion_refine_seed42.csv")

CANDIDATE_FIELDS = [
    "dataset",
    "variant",
    "seed",
    "status",
    "reason",
    "requested_ratio",
    "fullgraph_gate_passed",
    "promoted",
    "diagnostic_only",
    "invalid_reasons",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "total_condensed_node_ratio",
    "model_type",
    "feature_mode",
    "metapath_blocks",
    "path_lad_blocks",
    "loss_type",
    "hidden_dim",
    "dropout",
    "lr",
    "use_kd",
    "use_diffusion",
    "use_source_anchors",
    "use_coverage_medoid",
    "block_norm_stats_source",
    "teacher_val_acc",
    "kd_gate_passed",
    "kd_skip_reason",
    "source_log",
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_csv(path: str | Path) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _ratio_value(row: dict) -> float | None:
    return _float(row.get("requested_ratio", row.get("ratio")))


def _best_fullgraph(rows: list[dict], dataset: str) -> dict | None:
    candidates = []
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        acc = _float(row.get("accuracy"))
        if acc is not None:
            candidates.append((acc, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _gate_passed(rows: list[dict], dataset: str) -> bool:
    return any(row.get("dataset") == dataset and _truthy(row.get("gate_passed")) for row in rows)


def _write_report(rows: list[dict], path: Path, csv_path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| Dataset | Variant | Ratio | Acc | Macro-F1 | Status | Promoted | Reason |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset','')} | {row.get('variant','')} | {row.get('requested_ratio','')} | "
            f"{row.get('accuracy','')} | {row.get('macro_f1','')} | {row.get('status','')} | "
            f"{row.get('promoted','')} | {row.get('reason','')} |"
        )
    lines.extend(["", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_row(summary: dict, *, dataset: str, variant: str, seed: int, ratio: float | str, log_path: str | Path, fullgraph_gate_passed: bool, status: str | None = None, reason: str | None = None, diagnostic_only: bool = False, promoted: bool | None = None, hidden_dim: int | str = "", dropout: float | str = "", lr: float | str = "") -> dict:
    row = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "status": status or summary.get("status", "completed"),
        "reason": reason if reason is not None else summary.get("reason", ""),
        "requested_ratio": ratio,
        "fullgraph_gate_passed": fullgraph_gate_passed,
        "promoted": fullgraph_gate_passed and not diagnostic_only if promoted is None else promoted,
        "diagnostic_only": diagnostic_only,
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", ""),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", ""),
        "model_type": summary.get("model_type", ""),
        "feature_mode": summary.get("feature_mode", "schema_default_metapath"),
        "metapath_blocks": json.dumps(summary.get("metapath_blocks", [])),
        "path_lad_blocks": json.dumps(summary.get("path_lad_blocks", [])),
        "loss_type": summary.get("loss_type", ""),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "use_kd": summary.get("use_kd", False),
        "use_diffusion": summary.get("use_diffusion", summary.get("diffusion_enabled", False)),
        "use_source_anchors": summary.get("use_source_anchors", False),
        "use_coverage_medoid": summary.get("use_coverage_medoid", False),
        "block_norm_stats_source": summary.get("block_norm_stats_source", summary.get("compiled_block_stats_source", "")),
        "teacher_val_acc": summary.get("teacher_val_acc", ""),
        "kd_gate_passed": summary.get("kd_gate_passed", False),
        "kd_skip_reason": summary.get("kd_skip_reason", ""),
        "source_log": str(log_path),
    }
    if row["status"] != "completed":
        row["promoted"] = False
    checks = validate_promoted_row(row)
    row["invalid_reasons"] = json.dumps(checks["reasons"])
    if checks["reasons"]:
        row["promoted"] = False
    return row


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_acm_candidates(seed: int, epochs: int, fullgraph_gate_passed: bool, skip_existing: bool) -> list[dict]:
    rows: list[dict] = []
    log_dir = Path("experiments/logs/acm_s1_clean_tuned_seed42")
    log_dir.mkdir(parents=True, exist_ok=True)
    if not fullgraph_gate_passed:
        for ratio in [0.096, 0.12, 0.15]:
            rows.append(_candidate_row({}, dataset="acm", variant="S1_clean_metapath_sehgnn_tuned", seed=seed, ratio=ratio, log_path="", fullgraph_gate_passed=False, status="skipped_blocked_by_fullgraph_backbone", reason="ACM fullgraph gate did not pass", promoted=False))
        return rows
    graph = load_processed_small_dataset("acm")
    blocks, metadata = build_schema_default_blocks(graph, include_self=True, include_metapath=True)
    for ratio in [0.096, 0.12, 0.15]:
        for hidden_dim in [256, 512]:
            for dropout in [0.2, 0.3, 0.5]:
                for lr in [0.001, 0.003]:
                    for loss_type in ["clipped", "class_balanced"]:
                        variant = f"S1_clean_metapath_sehgnn_tuned_h{hidden_dim}_d{str(dropout).replace('.', 'p')}_lr{str(lr).replace('.', 'p')}_{loss_type}"
                        log_path = log_dir / f"acm_{variant}_r{str(ratio).replace('.', 'p')}_seed{seed}.json"
                        if skip_existing and log_path.exists():
                            summary = _load_json(log_path)
                        else:
                            try:
                                run = train_prototype_sehgnn_lite(
                                    graph,
                                    blocks=blocks,
                                    metadata=metadata,
                                    requested_ratio=ratio,
                                    seed=seed,
                                    epochs=epochs,
                                    hidden_dim=hidden_dim,
                                    dropout=dropout,
                                    lr=lr,
                                    weight_decay=1e-4,
                                    loss_type=loss_type,
                                    min_proto_per_class=4,
                                )
                                summary = {
                                    "dataset": "acm",
                                    "variant": variant,
                                    "seed": seed,
                                    "status": "completed",
                                    "target_type": graph.target_type,
                                    "feature_mode": "schema_default_metapath",
                                    "teacher_type": "none",
                                    "use_kd": False,
                                    "use_diffusion": False,
                                    "use_source_anchors": False,
                                    "use_coverage_medoid": False,
                                    **run.summary,
                                }
                            except Exception as exc:
                                summary = {
                                    "dataset": "acm",
                                    "variant": variant,
                                    "seed": seed,
                                    "status": "experiment_failed",
                                    "reason": str(exc),
                                    "use_diffusion": False,
                                }
                            write_json_summary(log_path, summary)
                        rows.append(_candidate_row(summary, dataset="acm", variant=variant, seed=seed, ratio=ratio, log_path=log_path, fullgraph_gate_passed=True, hidden_dim=hidden_dim, dropout=dropout, lr=lr))
    return rows


def _existing_small_rows(dataset: str, variants: list[str], ratios: list[float], *, seed: int, fullgraph_gate_passed: bool, reason: str) -> list[dict]:
    rows: list[dict] = []
    clean = _read_csv(SMALL_CLEAN_CSV)
    for ratio in ratios:
        for variant in variants:
            matches = [
                row for row in clean
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and (_ratio_value(row) is not None and abs(_ratio_value(row) - ratio) < 1e-9)
            ]
            source = matches[0] if matches else {}
            status = "diagnostic_existing" if matches else "missing_existing_diagnostic"
            if not fullgraph_gate_passed:
                status = "blocked_by_fullgraph_backbone"
            rows.append(_candidate_row(source, dataset=dataset, variant=variant, seed=seed, ratio=ratio, log_path=source.get("source_log", ""), fullgraph_gate_passed=fullgraph_gate_passed, status=status, reason=reason, diagnostic_only=True, promoted=False))
    return rows


def _medium_reference_rows(dataset: str, *, seed: int, fullgraph_gate_passed: bool, reason: str) -> list[dict]:
    rows: list[dict] = []
    medium = _read_csv(MEDIUM_CLEAN_CSV)
    variants = ["LAD_reference"]
    if dataset == "ogbn-arxiv":
        variants.extend(["LAD_reference_with_fixed_block_stats", "stronger_table_head"])
    else:
        variants.extend(["LAD_reference_balanced_softmax", "LAD_reference_logit_adjusted", "LAD_reference_label_smoothing", "stronger_table_head"])
    for ratio in [0.06, 0.12]:
        for variant in variants:
            if variant == "LAD_reference":
                matches = [
                    row for row in medium
                    if row.get("dataset") == dataset and row.get("variant") == "LAD_reference"
                    and (_ratio_value(row) is not None and abs(_ratio_value(row) - ratio) < 1e-9)
                ]
                source = matches[0] if matches else {}
                status = "diagnostic_existing" if matches else "missing_existing_diagnostic"
            else:
                source = {"model_type": "compiled_demand_mlp", "feature_mode": "label_affinity", "use_diffusion": False}
                status = "skipped_blocked_by_fullgraph_backbone" if not fullgraph_gate_passed else "skipped_resource_guard"
            row_reason = reason if status.startswith("skipped") or status.startswith("blocked") else "existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run"
            rows.append(_candidate_row(source, dataset=dataset, variant=variant, seed=seed, ratio=ratio, log_path=source.get("source_log", ""), fullgraph_gate_passed=fullgraph_gate_passed, status=status if not fullgraph_gate_passed or variant != "LAD_reference" else status, reason=row_reason, diagnostic_only=not fullgraph_gate_passed, promoted=fullgraph_gate_passed and variant == "LAD_reference" and status == "diagnostic_existing"))
    return rows


def _write_candidate_outputs(fullgraph_rows: list[dict], *, seed: int, epochs: int, skip_existing: bool) -> dict[str, list[dict]]:
    gates = {dataset: _gate_passed(fullgraph_rows, dataset) for dataset in ["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"]}
    outputs: dict[str, list[dict]] = {}

    outputs["acm"] = _run_acm_candidates(seed, epochs, gates["acm"], skip_existing)
    write_csv(ACM_CSV, outputs["acm"], CANDIDATE_FIELDS)
    _write_report(outputs["acm"], Path("experiments/reports/acm_s1_clean_tuned_seed42.md"), ACM_CSV, "ACM S1 Clean Tuned Seed 42")

    dblp_reason = "DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted"
    outputs["dblp"] = _existing_small_rows("dblp", ["S0_current_best", "S1_clean_APA_sehgnn"], [0.005, 0.065, 0.096], seed=seed, fullgraph_gate_passed=gates["dblp"], reason=dblp_reason)
    write_csv(DBLP_CSV, outputs["dblp"], CANDIDATE_FIELDS)
    _write_report(outputs["dblp"], Path("experiments/reports/dblp_schema_fixed_candidate_seed42.md"), DBLP_CSV, "DBLP Schema-First Candidate Seed 42")

    imdb_reason = "IMDB fullgraph gate did not pass; clean rows are diagnostics and Path-LAD/source-anchor paths are not promoted"
    outputs["imdb"] = _existing_small_rows("imdb", ["S1_clean_MAM_MDM_MKM"], [0.005, 0.025, 0.05], seed=seed, fullgraph_gate_passed=gates["imdb"], reason=imdb_reason)
    write_csv(IMDB_CSV, outputs["imdb"], CANDIDATE_FIELDS)
    _write_report(outputs["imdb"], Path("experiments/reports/imdb_fullgraph_first_candidate_seed42.md"), IMDB_CSV, "IMDB Fullgraph-First Candidate Seed 42")

    arxiv_reason = "ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic"
    outputs["ogbn-arxiv"] = _medium_reference_rows("ogbn-arxiv", seed=seed, fullgraph_gate_passed=gates["ogbn-arxiv"], reason=arxiv_reason)
    write_csv(ARXIV_CSV, outputs["ogbn-arxiv"], CANDIDATE_FIELDS)
    _write_report(outputs["ogbn-arxiv"], Path("experiments/reports/arxiv_no_diffusion_recovery_seed42.md"), ARXIV_CSV, "ogbn-arxiv No-Diffusion Recovery Seed 42")

    products_reason = "ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run"
    outputs["ogbn-products"] = _medium_reference_rows("ogbn-products", seed=seed, fullgraph_gate_passed=gates["ogbn-products"], reason=products_reason)
    write_csv(PRODUCTS_CSV, outputs["ogbn-products"], CANDIDATE_FIELDS)
    _write_report(outputs["ogbn-products"], Path("experiments/reports/products_no_diffusion_recovery_seed42.md"), PRODUCTS_CSV, "ogbn-products No-Diffusion Recovery Seed 42")

    return outputs


def _table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def _best_promoted(rows: list[dict]) -> list[dict]:
    by_dataset: dict[str, tuple[float, dict]] = {}
    for row in rows:
        if not _truthy(row.get("promoted")) or row.get("status") != "completed":
            continue
        if json.loads(row.get("invalid_reasons", "[]")):
            continue
        acc = _float(row.get("accuracy"))
        if acc is None:
            continue
        dataset = row.get("dataset", "")
        if dataset not in by_dataset or acc > by_dataset[dataset][0]:
            by_dataset[dataset] = (acc, row)
    return [item[1] for item in sorted(by_dataset.values(), key=lambda item: item[1].get("dataset", ""))]


def build_summary() -> None:
    schema = _read_csv(SCHEMA_CSV)
    fullgraph = _read_csv(FULLGRAPH_CSV)
    identity = _read_csv(IDENTITY_CSV)
    candidate_outputs = {
        "acm": _read_csv(ACM_CSV),
        "dblp": _read_csv(DBLP_CSV),
        "imdb": _read_csv(IMDB_CSV),
        "ogbn-arxiv": _read_csv(ARXIV_CSV),
        "ogbn-products": _read_csv(PRODUCTS_CSV),
    }
    all_candidates = [row for rows in candidate_outputs.values() for row in rows]
    promoted = _best_promoted(all_candidates)
    blocked = [row for row in all_candidates if not _truthy(row.get("promoted"))]
    fullgraph_decisions = []
    for dataset in ["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"]:
        best = _best_fullgraph(fullgraph, dataset) or {}
        fullgraph_decisions.append({
            "dataset": dataset,
            "best_variant": best.get("variant", ""),
            "accuracy": best.get("accuracy", ""),
            "gate": best.get("target_gate", ""),
            "gate_passed": best.get("gate_passed", ""),
            "blocked": best.get("blocked_by_fullgraph_backbone", ""),
        })

    lines = [
        "# Fullgraph Parity + Condensation Recovery Summary",
        "",
        "## 0. Stage Scope And Code Changes",
        "",
        "- Added fullgraph parity audit output with required hashes, split counts, schema counts, gate decisions, and resource fields.",
        "- Added schema alignment audit with `current_processed` versus `full_schema` loader modes; default condensation loading remains unchanged.",
        "- Added full-schema small-data loader entry point for alignment-only experiments while preserving the incoming-to-target default path.",
        "- Added identity/prototype/shadow gap decomposition with explicit compatibility flags so mismatched diagnostic rows are retained but not promoted.",
        "- Added explicit compiled block-stat APIs: `fit_block_stats`, `freeze_block_stats`, and `apply_block_stats`; stats are fit on original train target demand rows.",
        "- Added KD v2 gate/log helpers and tests; KD rows are skipped unless teacher quality logs pass the gate.",
        "- Added no-diffusion promoted-row validation so diffusion, products P2/two-hop LAD, CoverageMedoid, source anchors, and invalid KD rows are excluded from best summaries.",
        "- Added the stage runner and five dataset-specific candidate tables required by the prompt.",
        "",
        "## 1. Fullgraph Parity Status By Dataset",
        "",
        *_table(fullgraph_decisions, ["dataset", "best_variant", "accuracy", "gate", "gate_passed", "blocked"]),
        "",
        "## 2. Schema Completeness Status By Dataset",
        "",
        *_table(schema, ["dataset", "loader_name", "target_type", "metapath_available", "metapath_missing", "freehgc_or_hgb_alignment_status", "notes"]),
        "",
        "## 3. Identity Condensation Sanity Status",
        "",
        *_table(identity, ["dataset", "ratio", "fullgraph_acc", "identity_condensed_acc", "prototype_oracle_acc", "shadow_hgc_acc", "schema_compatible", "bottleneck_label"]),
        "",
        "## 4. Gap Decomposition Table",
        "",
        *_table(identity, ["dataset", "full_to_identity_gap", "identity_to_oracle_gap", "oracle_to_shadow_gap", "full_to_shadow_gap", "compatibility_reason"]),
        "",
        "## 5. Promoted Rows",
        "",
    ]
    if promoted:
        lines.extend(_table(promoted, ["dataset", "variant", "requested_ratio", "accuracy", "macro_f1", "status", "invalid_reasons"]))
    else:
        lines.append("No candidate row was promoted after fullgraph gates and validity checks.")
    lines.extend([
        "",
        "## 6. Blocked Rows And Reasons",
        "",
        *_table(blocked[:120], ["dataset", "variant", "requested_ratio", "status", "reason", "invalid_reasons"]),
        "",
        "## 7. Dropped Components",
        "",
        "- High-dimensional diffusion remains diagnostic-only and is not promoted.",
        "- Products P2 / two-hop LAD and products diffusion were not run in this stage.",
        "- CoverageMedoid and source anchors were not promoted.",
        "- Old KD rows are not promoted; KD v2 is skipped unless teacher quality logs pass the gate.",
        "- DBLP and IMDB condensation SOTA chasing is blocked while fullgraph/schema alignment remains below gate.",
        "",
        "## 8. Next-Stage Recommendation",
        "",
        "- ACM is the only dataset eligible for clean S1 tuning in this sprint if the fullgraph acceptable gate passes; use the best valid row from `acm_s1_clean_tuned_seed42.csv`.",
        "- DBLP needs schema/backbone alignment before condensation claims; full-schema loading is now auditable but the fullgraph gate remains the decision point.",
        "- IMDB needs a stronger aligned fullgraph backbone before Path-LAD/source-anchor or KD experiments are meaningful.",
        "- arxiv/products should stay no-diffusion; recover the fullgraph teacher ceiling before spending more runs on compressed variants.",
        "",
        "## 9. Acceptance Checklist",
        "",
        f"- Fullgraph parity table exists: `{FULLGRAPH_CSV.exists()}` (`{FULLGRAPH_CSV}`)",
        f"- Schema alignment table exists: `{SCHEMA_CSV.exists()}` (`{SCHEMA_CSV}`)",
        f"- Identity audit table exists: `{IDENTITY_CSV.exists()}` (`{IDENTITY_CSV}`)",
        f"- ACM candidate table exists: `{ACM_CSV.exists()}` (`{ACM_CSV}`)",
        f"- DBLP candidate table exists: `{DBLP_CSV.exists()}` (`{DBLP_CSV}`)",
        f"- IMDB candidate table exists: `{IMDB_CSV.exists()}` (`{IMDB_CSV}`)",
        f"- arxiv candidate table exists: `{ARXIV_CSV.exists()}` (`{ARXIV_CSV}`)",
        f"- products candidate table exists: `{PRODUCTS_CSV.exists()}` (`{PRODUCTS_CSV}`)",
        "- Invalid rows are retained in artifacts but excluded from promoted best-row summaries.",
        "- KD v2 is skipped unless teacher gate passes.",
        "",
        "## Files",
        "",
        f"- Schema audit: `{SCHEMA_CSV}`",
        f"- Fullgraph parity: `{FULLGRAPH_CSV}`",
        f"- Identity audit: `{IDENTITY_CSV}`",
        f"- ACM candidates: `{ACM_CSV}`",
        f"- DBLP candidates: `{DBLP_CSV}`",
        f"- IMDB candidates: `{IMDB_CSV}`",
        f"- arxiv candidates: `{ARXIV_CSV}`",
        f"- products candidates: `{PRODUCTS_CSV}`",
    ])
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fullgraph parity + condensation recovery stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-summary", action="store_true")
    args = parser.parse_args()

    if not args.only_summary:
        base = [sys.executable]
        _run(base + ["scripts/run_schema_alignment_audit.py", "--seed", str(args.seed)])
        _run(base + ["scripts/run_fullgraph_parity.py", "--seed", str(args.seed), "--epochs", str(args.epochs), "--skip-existing"])
        _run(base + ["scripts/run_identity_condensation_audit.py", "--seed", str(args.seed)])
        fullgraph_rows = _read_csv(FULLGRAPH_CSV)
        _write_candidate_outputs(fullgraph_rows, seed=args.seed, epochs=args.epochs, skip_existing=args.skip_existing)
    build_summary()


if __name__ == "__main__":
    main()
