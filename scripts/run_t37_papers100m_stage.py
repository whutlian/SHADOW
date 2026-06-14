from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.ultra.papers100m_t37_contract import summarize_t37_stage


def _top(rows: list[dict[str, str]], n: int = 12) -> list[dict[str, str]]:
    return sorted([row for row in rows if row.get("accuracy", "") != ""], key=lambda row: float(row.get("accuracy", 0.0) or 0.0), reverse=True)[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T37 papers100M stage summary.")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--summaries-dir", default="experiments/summaries")
    parser.add_argument("--out", default="experiments/tables/t37_papers100m_stage_summary.csv")
    args = parser.parse_args()

    tables = Path(args.tables_dir)
    summaries = Path(args.summaries_dir)
    bank = read_csv(tables / "t37_papers100m_scr_bank_audit.csv")
    disco = read_csv(tables / "t37_papers100m_disco_parity_scr_seed42.csv")
    native = read_csv(tables / "t37_papers100m_native_randcore_seed42.csv")
    disco_multi = read_csv(tables / "t37_papers100m_disco_parity_scr_multiseed.csv")
    native_multi = read_csv(tables / "t37_papers100m_native_randcore_multiseed.csv")
    teacher = read_csv(tables / "t37_papers100m_teacher_light_upgrade.csv")
    summary = summarize_t37_stage(disco_rows=disco, native_rows=native, multiseed_rows=disco_multi + native_multi, bank_rows=bank, teacher_rows=teacher)
    write_csv(args.out, [summary])
    lines = [
        "# T37 Papers100M SCR / STT-RandCore Summary",
        "",
        "## Files Changed",
        "",
        "- Added T37 contract, SCR bank/rank builder, SCR prefix audit, hard-label SGC parity support, and T37 runner scripts.",
        "- Added T37 tests for row guards, relative metrics, SCR class floors, deterministic rank, teacher weighting caps, prefix nesting, and no valid/test label leakage.",
        "",
        "## Stage Summary",
        "",
        *markdown_table([summary], list(summary.keys())),
        "",
        "## SCR Bank Audit",
        "",
        *markdown_table(bank, ["method", "seed", "requested_full_node_ratio", "selected_count", "selected_class_count", "class_floor_actual_min", "prefix_violation_count", "coverage_bucket_count", "selected_hard_label_prior_kl"]),
        "",
        "## DisCo-Parity Seed42",
        "",
        *markdown_table(disco, ["method", "backend", "requested_full_node_ratio", "accuracy", "macro_f1", "disco_acc", "random_onecache_acc", "beats_disco", "beats_random_onecache", "promotion_status", "failure_reason"]),
        "",
        "## Native RandCore Seed42",
        "",
        *markdown_table(native, ["method", "backend", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "promotion_status", "failure_reason", "notes"]),
        "",
        "## Multi-Seed Aggregate",
        "",
        *markdown_table(disco_multi + native_multi, ["method", "backend", "requested_full_node_ratio", "accuracy", "macro_f1", "notes"]),
        "",
        "## Current Best Rows",
        "",
        *markdown_table(_top(disco + native), ["method", "backend", "requested_full_node_ratio", "valid_acc", "accuracy", "macro_f1", "promotion_status"]),
        "",
        "## Recommendation",
        "",
        "- If `stop_condition_met=True`, frame papers100M as scalability demonstration plus one-cache multi-ratio evidence rather than a method-SOTA battleground.",
        "- If `stop_condition_met=False`, SCR/STT-RandCore is worth one follow-up only if the winning rows also pass all forbidden guards.",
    ]
    ensure_report(summaries / "t37_papers100m_scr_stage_summary.md", lines)
    print("status=completed")


if __name__ == "__main__":
    main()
