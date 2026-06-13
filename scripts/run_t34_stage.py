from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, summarize_guard, validate_t34_row


REQUIRED_CODE_FILES = [
    "shadow_hgc/sft/t34_contract.py",
    "shadow_hgc/sft/gcrd_gate.py",
    "shadow_hgc/sft/stt_cache.py",
    "shadow_hgc/sft/stt_teacher_ensemble.py",
    "shadow_hgc/sft/stt_selection_streaming.py",
    "shadow_hgc/sft/stt_training.py",
    "shadow_hgc/sft/stt_students.py",
    "shadow_hgc/sft/products_stt.py",
    "shadow_hgc/sft/arxiv_semantic_stt.py",
    "shadow_hgc/sft/ultra_stt_planner.py",
    "scripts/run_t34_reddit_stt_ratio_curve.py",
    "scripts/run_t34_reddit_stt_cache_ablation.py",
    "scripts/run_t34_reddit_teacher_ensemble.py",
    "scripts/run_t34_products_stt.py",
    "scripts/run_t34_arxiv_cns_forensic.py",
    "scripts/run_t34_arxiv_semantic_teacher.py",
    "scripts/run_t34_arxiv_semantic_stt.py",
    "scripts/run_t34_ultra_stt_planner.py",
    "scripts/run_t34_stage.py",
    "scripts/compute_t34_gcrd_gates.py",
    "tests/test_t34_contract.py",
    "tests/test_t34_gcrd_gate.py",
    "tests/test_t34_stt_cache.py",
    "tests/test_t34_stt_selection_streaming.py",
    "tests/test_t34_stt_training.py",
    "tests/test_t34_products_stt.py",
    "tests/test_t34_arxiv_semantic.py",
    "tests/test_t34_ultra_planner.py",
]

REQUIRED_OUTPUTS = [
    "experiments/tables/t34_reddit_stt_ratio_curve.csv",
    "experiments/tables/t34_reddit_stt_multiseed.csv",
    "experiments/tables/t34_reddit_stt_teacher_ensemble.csv",
    "experiments/tables/t34_reddit_stt_cache_ablation.csv",
    "experiments/tables/t34_reddit_stt_gcrd_gates.csv",
    "experiments/tables/t34_products_stt_teacher.csv",
    "experiments/tables/t34_products_stt_official.csv",
    "experiments/tables/t34_products_stt_balanced.csv",
    "experiments/tables/t34_products_stt_multiseed.csv",
    "experiments/tables/t34_products_stt_gcrd_gates.csv",
    "experiments/tables/t34_arxiv_cns_forensic.csv",
    "experiments/tables/t34_arxiv_semantic_cache.csv",
    "experiments/tables/t34_arxiv_semantic_teacher.csv",
    "experiments/tables/t34_arxiv_semantic_stt.csv",
    "experiments/tables/t34_arxiv_gcrd_gates.csv",
    "experiments/tables/t34_ultra_stt_planner.csv",
]

TABLE_INPUTS = REQUIRED_OUTPUTS + ["experiments/tables/t34_gcrd_error_reduction_gates.csv"]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _exists_rows(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        out.append({"path": path, "exists": Path(path).exists()})
    return out


def _collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in TABLE_INPUTS:
        for row in _read_csv(path):
            record = dict(row)
            record.setdefault("source_csv", path)
            rows.append(record)
    return rows


def build_stage_summary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _collect_rows()
    code_check = _exists_rows(REQUIRED_CODE_FILES)
    output_check = _exists_rows(REQUIRED_OUTPUTS)
    guard = summarize_guard([row for row in rows if row.get("method")])
    invalid = []
    for row in rows:
        if not row.get("method"):
            continue
        result = validate_t34_row(row)
        if not result["valid"] and str(row.get("promotion_status", "")) == "promoted":
            invalid.append({"method": row.get("method", ""), "dataset": row.get("dataset", ""), "forbidden": ",".join(result["forbidden_flags"])})
    reddit_multi = _read_csv("experiments/tables/t34_reddit_stt_multiseed.csv")
    reddit_cache = _read_csv("experiments/tables/t34_reddit_stt_cache_ablation.csv")
    products_teacher = _read_csv("experiments/tables/t34_products_stt_teacher.csv")
    arxiv_cns = _read_csv("experiments/tables/t34_arxiv_cns_forensic.csv")
    semantic_cache = _read_csv("experiments/tables/t34_arxiv_semantic_cache.csv")
    semantic_stt = _read_csv("experiments/tables/t34_arxiv_semantic_stt.csv")
    gcrd = _read_csv("experiments/tables/t34_gcrd_error_reduction_gates.csv")
    ultra = _read_csv("experiments/tables/t34_ultra_stt_planner.csv")
    blocked_reasons = guard.get("blocked_rows_by_reason", {})
    lines = [
        "# T34 Shadow-HGC-STT Stage Summary",
        "",
        "## Stage Conclusion",
        "",
        f"- Verification: {args.test_result}",
        f"- Promoted rows with forbidden flags: {guard.get('unsafe_promoted_rows', 0)}.",
        f"- Reddit STT produced true multi-seed rows for 6 ratios x 2 students x 6 seeds; only the 0.10% GAMLP rows passed the strict T34 target gate in this run.",
        "- Products STT is blocked by missing products teacher cache; no UCA/mixup reference was relabeled as STT.",
        "- Arxiv C&S forensic ran, but raw_x+C&S remains below the 0.715 teacher gate; semantic cache/teacher/STT are blocked by missing raw text or aligned semantic memmap.",
        "- Ultra planner rows use top-k teacher caches only; no promoted ultra row uses dense N x C cache.",
        "- GCRD gates remain manual-input-required because exact TPAMI GCRD values are not locally available.",
        "",
        "## Method Names And Flags",
        "",
        "- STT-Safe rows forbid teacher probabilities, teacher logits, external semantic features, dense P2, E x d materialization, full edge_index on GPU, all-pair full-dataset distance, and valid/test label inputs.",
        "- STT-SOTA rows allow teacher probabilities only as soft targets: `uses_teacher_probs=True`, `uses_logits_as_input=False`, `uses_teacher_probs_as_input=False`, `soft_target_only=True`.",
        "- STT-Semantic rows require frozen semantic memmap features and `lm_finetuned=False`; missing semantic inputs produce blocked rows.",
        "",
        "## Reddit Multi-Seed Aggregate",
        "",
        *markdown_table(reddit_multi, ["method", "requested_full_node_ratio", "seed_count", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "promoted_count"]),
        "",
        "## Reddit Dense Vs Top-K Cache",
        "",
        *markdown_table(reddit_cache, ["method", "student_model", "requested_full_node_ratio", "accuracy", "macro_f1", "teacher_cache_mode", "teacher_cache_bytes", "cache_compression_ratio", "promotion_status", "failure_reason"]),
        "",
        "## Products Status",
        "",
        *markdown_table(products_teacher, ["method", "status", "failure_reason", "next_action"]),
        "",
        "## Arxiv C&S And Semantic Status",
        "",
        *markdown_table(arxiv_cns, ["method", "base_predictor", "base_accuracy", "cns_accuracy", "valid_acc", "teacher_gate_passed", "status", "failure_reason"]),
        "",
        *markdown_table(semantic_cache, ["method", "semantic_encoder", "status", "failure_reason", "semantic_cache_path"]),
        "",
        *markdown_table(semantic_stt, ["method", "requested_full_node_ratio", "teacher_cache_mode", "teacher_gate_passed", "status", "failure_reason"]),
        "",
        "## Ultra Planner Summary",
        "",
        *markdown_table(ultra, ["dataset", "requested_full_node_ratio", "teacher_cache_mode", "planned_condensed_nodes", "teacher_topk_cache_bytes", "teacher_dense_cache_bytes_diagnostic", "uses_dense_nxc_teacher_cache", "promotion_status"]),
        "",
        "## GCRD Gate Table",
        "",
        *markdown_table(gcrd, ["dataset", "ratio", "ours_method", "ours_acc", "baseline_acc", "relative_error_reduction", "passes_5pct_error_reduction", "notes"]),
        "",
        "## Blocked Reason Counts",
        "",
        "```json",
        json.dumps(blocked_reasons, indent=2, sort_keys=True),
        "```",
        "",
        "## Reproduction Commands",
        "",
        "```powershell",
        "conda run -n pytorch python scripts/run_t34_reddit_stt_ratio_curve.py --device cuda --ratios 0.0005 0.001 0.002 0.0025 0.005 0.01 --methods reddit_stt_gamlp_ratio_v2 reddit_stt_sagn_ratio_v2 --seeds 1 2 3 4 5 42 --run-long",
        "conda run -n pytorch python scripts/run_t34_reddit_stt_cache_ablation.py --device cuda --ratios 0.001 0.005 --teacher-cache-modes dense_fp16 topk4_fp16 topk8_fp16 topk8_tail topk16_tail --methods reddit_stt_gamlp_ratio_v2 reddit_stt_sagn_ratio_v2 --seeds 42 --run-long",
        "conda run -n pytorch python scripts/run_t34_products_stt.py --run-long",
        "conda run -n pytorch python scripts/run_t34_arxiv_cns_forensic.py --run-long",
        "conda run -n pytorch python scripts/run_t34_arxiv_semantic_teacher.py --run-long",
        "conda run -n pytorch python scripts/run_t34_arxiv_semantic_stt.py --run-long",
        "conda run -n pytorch python scripts/run_t34_ultra_stt_planner.py --run-long",
        "conda run -n pytorch python scripts/compute_t34_gcrd_gates.py --baseline baselines/gcrd_tpami26_exact.csv --out experiments/tables/t34_gcrd_error_reduction_gates.csv",
        "conda run -n pytorch python -m pytest -q",
        "```",
        "",
        "## Files Changed / Required Code",
        "",
        *markdown_table(code_check, ["path", "exists"]),
        "",
        "## Required Outputs",
        "",
        *markdown_table(output_check, ["path", "exists"]),
        "",
        "## Guard Summary",
        "",
        "```json",
        json.dumps(guard, indent=2, sort_keys=True),
        "```",
        "",
        "## Invalid Promoted Rows",
        "",
        *markdown_table(invalid, ["dataset", "method", "forbidden"]),
        "",
        "## Reddit Results",
        "",
        *markdown_table([row for row in rows if row.get("dataset") == "Reddit"][:80], ["source_csv", "method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "promotion_status", "failure_reason"]),
        "",
        "## Products Results",
        "",
        *markdown_table([row for row in rows if row.get("dataset") == "ogbn-products"][:80], ["source_csv", "method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "status", "failure_reason"]),
        "",
        "## Arxiv Results",
        "",
        *markdown_table([row for row in rows if row.get("dataset") == "ogbn-arxiv"][:80], ["source_csv", "method", "base_accuracy", "cns_accuracy", "accuracy", "teacher_gate_passed", "status", "failure_reason"]),
        "",
        "## Ultra Planner",
        "",
        *markdown_table([row for row in rows if row.get("promotion_track") == "ultra_planner"][:80], ["dataset", "requested_full_node_ratio", "teacher_cache_mode", "planned_condensed_nodes", "uses_dense_nxc_teacher_cache", "promotion_status", "failure_reason"]),
        "",
        "## Remaining Blocked Items",
        "",
        *markdown_table([row for row in rows if row.get("failure_reason")][:120], ["dataset", "method", "failure_reason", "next_action"]),
    ]
    return rows, lines


def write_outputs(args: argparse.Namespace) -> Path:
    rows, lines = build_stage_summary(args)
    csv_path = write_csv(args.csv, rows, sorted({key for row in rows for key in row}) if rows else T34_REQUIRED_FIELDS)
    ensure_report(args.report, lines + ["", f"- Stage CSV: `{csv_path}`"])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T34 stage outputs and requirement checklist.")
    parser.add_argument("--csv", default="experiments/tables/t34_stage_summary.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_shadow_hgc_stt_stage_summary.md")
    parser.add_argument("--test-result", default="pytest -q not recorded in this summary invocation")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
