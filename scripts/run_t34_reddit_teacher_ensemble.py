from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, make_t34_row


def build_teacher_ensemble_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    cache_dir = Path(args.teacher_ensemble_cache_dir)
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return [
            make_t34_row(
                dataset="Reddit",
                method="reddit_stt_teacher_ensemble",
                seed=int(args.seed),
                status="blocked",
                failure_reason="missing_reddit_teacher_ensemble_cache",
                promotion_track="sota_chase",
                promotion_status="not_promoted",
                uses_teacher_probs=True,
                uses_teacher_logits=True,
                soft_target_only=True,
                next_action="train independent teacher pool listed in T34 prompt",
            )
        ]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    teachers = meta.get("teachers", [])
    diag = meta.get("diagnostics", {})
    pairwise = float(diag.get("teacher_pairwise_kl_mean", 0.0) or 0.0)
    disagreement = float(diag.get("teacher_disagreement_mean", 0.0) or 0.0)
    diversity_failed = len(teachers) < 2 or pairwise <= 1e-6 or disagreement <= 1e-6
    row = make_t34_row(
        dataset="Reddit",
        method="reddit_stt_teacher_ensemble",
        seed=int(args.seed),
        status="completed_diagnostic",
        failure_reason="teacher_ensemble_diversity_failed" if diversity_failed else "",
        promotion_track="sota_chase",
        promotion_status="not_promoted",
        teacher_method="ensemble",
        teacher_cache_mode="dense_fp16",
        teacher_cache_bytes=(cache_dir / "teacher_probs.npy").stat().st_size if (cache_dir / "teacher_probs.npy").exists() else "",
        uses_teacher_probs=True,
        uses_teacher_logits=True,
        uses_logits_as_input=False,
        uses_teacher_probs_as_input=False,
        soft_target_only=True,
        teacher_ensemble_size=len(teachers),
        teacher_accuracy_each=json.dumps([t.get("teacher_accuracy", "") for t in teachers]),
        teacher_valid_acc_each=json.dumps([t.get("teacher_valid_acc", "") for t in teachers]),
        teacher_entropy_each=json.dumps([t.get("teacher_entropy_mean", "") for t in teachers]),
        teacher_pairwise_kl_mean=diag.get("teacher_pairwise_kl_mean", 0.0),
        teacher_pairwise_kl_min=diag.get("teacher_pairwise_kl_min", 0.0),
        teacher_pairwise_kl_max=diag.get("teacher_pairwise_kl_max", 0.0),
        teacher_disagreement_mean=diag.get("teacher_disagreement_mean", 0.0),
        teacher_ensemble_diversity_failed=diversity_failed,
        teacher_temperature=teachers[0].get("teacher_temperature", "") if teachers else "",
        calibrated_ensemble_accuracy=diag.get("calibrated_ensemble_accuracy", ""),
        oracle_ensemble_accuracy_if_available=diag.get("oracle_ensemble_accuracy_if_available", ""),
        valid_ECE=diag.get("valid_ECE", ""),
        valid_NLL=diag.get("valid_NLL", ""),
        notes=f"metadata={meta_path}",
    )
    return [row]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_teacher_ensemble_rows(args)
    csv_path = write_csv(args.csv, rows, T34_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        ["# T34 Reddit Teacher Ensemble", "", *markdown_table(rows, ["method", "teacher_ensemble_size", "teacher_pairwise_kl_mean", "teacher_disagreement_mean", "teacher_ensemble_diversity_failed", "failure_reason"]), "", f"- CSV: `{csv_path}`"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 Reddit teacher ensemble diagnostic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-ensemble-cache-dir", default="experiments/cache/t32_reddit_teacher_ensemble_seed42")
    parser.add_argument("--csv", default="experiments/tables/t34_reddit_stt_teacher_ensemble.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_reddit_teacher_ensemble.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
