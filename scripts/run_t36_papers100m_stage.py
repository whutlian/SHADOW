from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.ultra.papers100m_disco_parity import ensure_disco_baseline_csv
from shadow_hgc.ultra.papers100m_memmap import read_json
from shadow_hgc.ultra.papers100m_t36_contract import summarize_stage_gates, validate_t36_row


def _read(path: Path) -> list[dict[str, Any]]:
    return list(read_csv(path))


def _top_rows(rows: list[dict[str, Any]], n: int = 12) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (float(row.get("accuracy", 0.0) or 0.0), float(row.get("valid_acc", 0.0) or 0.0))

    return sorted(rows, key=key, reverse=True)[:n]


def _write_notes(summaries_dir: Path, name: str, title: str, rows: list[dict[str, Any]], fields: list[str], extra: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    if extra:
        lines.extend(extra)
        lines.append("")
    lines.extend(markdown_table(rows, fields))
    ensure_report(summaries_dir / name, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="T36 papers100M stage aggregator.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--baseline-csv", default="baselines/disco_papers100m_sgc.csv")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--summaries-dir", default="experiments/summaries")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    tables_dir = Path(args.tables_dir)
    summaries_dir = Path(args.summaries_dir)
    ensure_disco_baseline_csv(args.baseline_csv)
    teacher = _read(tables_dir / "t36_papers100m_teacher_upgrade.csv")
    nested = _read(tables_dir / "t36_papers100m_nested_bank_audit.csv")
    disco = _read(tables_dir / "t36_papers100m_disco_parity.csv")
    external = _read(tables_dir / "t36_papers100m_external_baselines.csv")
    ant = _read(tables_dir / "t36_papers100m_ant.csv")
    scale = _read(tables_dir / "t36_papers100m_scale_fidelity.csv")
    gates = summarize_stage_gates(teacher_rows=teacher, nested_rows=nested, disco_rows=disco, ant_rows=ant, scale_rows=scale)
    write_csv(tables_dir / "t36_papers100m_stage_summary.csv", [gates])

    manifest = read_json(cache_root / "manifest.json") if (cache_root / "manifest.json").exists() else {}
    forbidden = [validate_t36_row(row) for row in disco + ant + scale]
    unsafe = sum(1 for item in forbidden if not item["valid"])
    lines = [
        "# T36 Papers100M DisCo-Parity + Teacher/Bank/ANT Summary",
        "",
        "## Files Changed",
        "",
        "- Added T36 contract, teacher upgrade, nested bank, ratio policy, SGC backend, ANT translator, and DisCo parity modules.",
        "- Added T36 runner scripts for teacher, nested bank, DisCo parity, external baselines, ANT, scale fidelity, and stage aggregation.",
        "- Added toy tests for ratio conversion, baseline loading, nested prefixes, SGC scatter-add, ANT boundedness, and forbidden promoted paths.",
        "",
        "## Cache Context",
        "",
        *markdown_table([manifest] if manifest else [], ["dataset_name", "num_nodes", "num_edges", "feature_dim", "num_classes", "train_size", "valid_size", "test_size", "target_universe_size", "cache_build_id"]),
        "",
        "## Stage Gates",
        "",
        *markdown_table([gates], list(gates.keys())),
        "",
        "## Teacher Upgrade Results",
        "",
        *markdown_table(teacher, ["method", "feature_block_mode", "teacher_cache_mode", "valid_acc", "test_acc", "macro_f1", "predicted_classes", "promotion_status", "failure_reason"]),
        "",
        "## Nested Bank Audit",
        "",
        *markdown_table(nested, ["ratio", "selected_count", "prefix_overlap_with_previous_ratio", "prefix_violation_count", "selected_predicted_class_count", "selected_train_anchor_count", "selected_soft_prior_kl"]),
        "",
        "## DisCo-Parity Rows",
        "",
        *markdown_table(disco, ["method", "backend", "requested_full_node_ratio", "condensed_nodes", "valid_acc", "accuracy", "macro_f1", "disco_acc", "beats_disco", "promotion_status", "failure_reason"]),
        "",
        "## External Baselines",
        "",
        *markdown_table(external, ["method", "backend", "requested_full_node_ratio", "accuracy", "random_acc_baseline", "herding_acc_baseline", "kcenter_acc_baseline", "promotion_status", "failure_reason"]),
        "",
        "## ANT Results",
        "",
        *markdown_table(ant, ["method", "backend", "requested_full_node_ratio", "ant_edge_topk", "ant_edges", "ant_candidate_count", "accuracy", "promotion_status", "failure_reason"]),
        "",
        "## Scale-Fidelity Results",
        "",
        *markdown_table(scale, ["method", "backend", "requested_full_node_ratio", "condensed_nodes", "valid_acc", "accuracy", "macro_f1", "promotion_status", "failure_reason"]),
        "",
        "## Forbidden-Path Guard Summary",
        "",
        f"- Unsafe promoted/checked rows: {unsafe}",
        "- Promoted rows require no dense teacher RAM cache, no all-node dense teacher cache, no full edge GPU path, no E x d materialization, no dense P2, no exact all-pair distance, and no valid/test labels as inputs.",
        "",
        "## Current Best Rows",
        "",
        *markdown_table(_top_rows(disco + ant + scale), ["method", "backend", "requested_full_node_ratio", "valid_acc", "accuracy", "macro_f1", "promotion_status"]),
    ]
    ensure_report(summaries_dir / "t36_papers100m_stage_summary.md", lines)
    _write_notes(
        summaries_dir,
        "t36_disco_parity_notes.md",
        "T36 DisCo-Parity Notes",
        disco,
        ["method", "backend", "requested_full_node_ratio", "accuracy", "disco_acc", "absolute_gain_vs_disco", "beats_disco", "promotion_status", "failure_reason"],
        ["Native table results are not DisCo-parity. DisCo-parity rows require backend=SGC and the four exact low ratios."],
    )
    _write_notes(
        summaries_dir,
        "t36_nested_bank_notes.md",
        "T36 Nested Bank Notes",
        nested,
        ["ratio", "selected_count", "prefix_violation_count", "selected_predicted_class_count", "selected_train_anchor_count", "selected_bucket_histogram_json"],
    )
    _write_notes(
        summaries_dir,
        "t36_ant_notes.md",
        "T36 ANT Notes",
        ant,
        ["method", "backend", "requested_full_node_ratio", "ant_edge_topk", "ant_edges", "ant_candidate_count", "accuracy", "promotion_status", "failure_reason"],
        ["ANT candidate generation is bounded by ratio prefix size, edge_topk, and candidate_multiplier; no all-pair M^2 scoring is used."],
    )
    print("status=completed")


if __name__ == "__main__":
    main()
