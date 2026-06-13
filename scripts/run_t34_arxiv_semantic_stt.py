from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.stt_cache import estimate_stt_cache_bytes
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, make_t34_row


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _teacher_gate_passed(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        try:
            acc = float(row.get("accuracy", "") or row.get("teacher_accuracy", "") or 0.0)
        except ValueError:
            acc = 0.0
        if acc >= 0.740 or str(row.get("teacher_gate_passed", "")).lower() == "true":
            return True
    return False


def build_semantic_stt_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    teacher_rows = _read_csv(args.teacher_csv)
    gate = _teacher_gate_passed(teacher_rows)
    rows: list[dict[str, Any]] = []
    specs = [
        ("arxiv_semantic_stt_0p25", 0.0025, "dense_fp16"),
        ("arxiv_semantic_stt_0p50", 0.005, "dense_fp16"),
        ("arxiv_semantic_stt_topk8_tail_0p50", 0.005, "topk8_tail"),
        ("arxiv_semantic_stt_topk16_tail_0p50", 0.005, "topk16_tail"),
    ]
    for method, ratio, mode in specs:
        estimates = estimate_stt_cache_bytes(num_nodes=169_343, num_classes=40, mode=mode)
        rows.append(
            make_t34_row(
                dataset="ogbn-arxiv",
                method=method,
                seed=int(args.seed),
                requested_full_node_ratio=ratio,
                status="blocked",
                failure_reason="semantic_teacher_gate_not_passed" if not gate else "semantic_stt_training_not_implemented",
                promotion_track="semantic_sota",
                promotion_status="not_promoted",
                teacher_gate_passed=gate,
                uses_teacher_probs=True,
                uses_teacher_logits=True,
                uses_logits_as_input=False,
                uses_teacher_probs_as_input=False,
                uses_external_text_features=True,
                semantic_features_are_frozen=gate,
                lm_finetuned=False,
                semantic_cache_memmap=gate,
                soft_target_only=True,
                teacher_cache_mode=mode,
                teacher_cache_bytes=estimates["teacher_cache_bytes"],
                teacher_dense_cache_bytes_diagnostic=estimates["teacher_dense_cache_bytes_diagnostic"],
                cache_compression_ratio=estimates["cache_compression_ratio"],
                next_action="run semantic teacher first; condensation is gated at teacher >=0.740",
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_semantic_stt_rows(args)
    csv_path = write_csv(args.csv, rows, T34_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        ["# T34 Arxiv Semantic STT", "", *markdown_table(rows, ["method", "requested_full_node_ratio", "teacher_cache_mode", "teacher_gate_passed", "status", "failure_reason"]), "", f"- CSV: `{csv_path}`"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 arxiv semantic STT gated condensation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-csv", default="experiments/tables/t34_arxiv_semantic_teacher.csv")
    parser.add_argument("--csv", default="experiments/tables/t34_arxiv_semantic_stt.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_arxiv_semantic_stt_summary.md")
    parser.add_argument("--run-long", action="store_true")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
