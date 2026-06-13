from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t33_arxiv_cns_forensic import build_arxiv_forensic_rows as build_t33_arxiv_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, apply_t34_promotion_guard, make_t34_row


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _method(name: str) -> str:
    if name == "arxiv_raw_x_mlp_cns_forensic_v4":
        return "arxiv_raw_x_mlp_cns_forensic_v5"
    if name == "arxiv_raw_x_mlp_base_v4":
        return "arxiv_raw_x_mlp_base_v5"
    return name.replace("_cns_forensic_v4", "_cns_forensic_v5")


def t33_to_t34_arxiv(row: dict[str, Any]) -> dict[str, Any]:
    cns_acc = row.get("cns_accuracy", row.get("accuracy", ""))
    teacher_gate = cns_acc not in {"", None} and _f(cns_acc) >= 0.715
    reason = str(row.get("failure_reason", ""))
    if not teacher_gate and str(row.get("method", "")).endswith("cns_forensic_v4") and not reason:
        reason = "arxiv_teacher_gate_0p715_not_passed"
    out = make_t34_row(
        dataset="ogbn-arxiv",
        method=_method(str(row.get("method", ""))),
        seed=int(_f(row.get("seed", 42), 42)),
        accuracy=row.get("accuracy", ""),
        macro_f1=row.get("macro_f1", ""),
        valid_acc=row.get("valid_acc", ""),
        predicted_classes=row.get("predicted_classes", ""),
        status=row.get("status", "blocked"),
        failure_reason=reason,
        promotion_track="safe_main",
        promotion_status="promoted" if teacher_gate else "not_promoted",
        base_predictor=row.get("base_predictor", ""),
        base_accuracy=row.get("base_accuracy", ""),
        cns_accuracy=cns_acc,
        graph_direction=row.get("graph_direction", ""),
        normalization_mode=row.get("normalization_mode", ""),
        self_loop_mode=row.get("self_loop_mode", ""),
        logits_cache_hash=row.get("logits_cache_hash", ""),
        feature_checksum=row.get("feature_checksum", ""),
        edge_checksum=row.get("edge_checksum", ""),
        mask_checksum=row.get("mask_checksum", ""),
        teacher_gate_passed=teacher_gate,
        uses_teacher_probs=False,
        uses_teacher_logits=False,
        uses_external_text_features=False,
        notes=row.get("notes", ""),
    )
    return apply_t34_promotion_guard(out)


def build_arxiv_cns_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [t33_to_t34_arxiv(row) for row in build_t33_arxiv_rows(args)]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_cns_rows(args)
    csv_path = write_csv(args.csv, rows, T34_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        ["# T34 Arxiv C&S Forensic", "", *markdown_table(rows, ["method", "base_predictor", "base_accuracy", "cns_accuracy", "valid_acc", "graph_direction", "teacher_gate_passed", "status", "failure_reason"]), "", f"- CSV: `{csv_path}`"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 arxiv C&S forensic v5.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--base-logits-dir", default="experiments/logits/t33_arxiv")
    parser.add_argument("--fallback-t31-logits-dir", default="experiments/logits/t31_arxiv")
    parser.add_argument("--base-predictors", nargs="+", default=["raw_x_mlp", "mlp_on_sft", "sagn_lite_v5", "gamlp_lite_v5"])
    parser.add_argument("--train-base-logits-if-missing", action="store_true")
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--graph-directions", nargs="+", default=["cite_ref", "cited_by", "undirected_sym"])
    parser.add_argument("--normalization-modes", nargs="+", default=["dst_row"])
    parser.add_argument("--self-loop-modes", nargs="+", default=["none"])
    parser.add_argument("--correction-alphas", nargs="+", type=float, default=[0.2])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--correction-steps", nargs="+", type=int, default=[10])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--autoscale", nargs="+", default=["off"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t34_arxiv_cns_forensic.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_arxiv_cns_forensic.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
