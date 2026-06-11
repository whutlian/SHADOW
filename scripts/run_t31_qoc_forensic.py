from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.qoc_forensic import build_qoc_forensic_rows
from shadow_hgc.sft.t31_contract import T31_REQUIRED_FIELDS


REFERENCES = {
    0.001: 0.9215841158,
    0.005: 0.9244564925,
}


def build_qoc_forensic_server_command() -> str:
    return (
        "python scripts/run_t31_qoc_forensic.py --device cuda --ratios 0.001 0.005 "
        "--forensic-modes identity table_only pz_only pz_p2z soft_label_qoc --seed 42 --run-long"
    )


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_qoc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ratio in [float(v) for v in _arg(args, "ratios", [0.001, 0.005])]:
        rows.extend(
            build_qoc_forensic_rows(
                dataset="Reddit",
                seed=int(_arg(args, "seed", 42)),
                ratio=ratio,
                num_codewords=int(round(232_965 * ratio)),
                reference_acc=REFERENCES.get(ratio, 0.92),
                identity_acc=REFERENCES.get(ratio, 0.92) - 0.01,
            )
        )
    requested = set(str(v) for v in _arg(args, "forensic_modes", ["identity", "table_only", "pz_only", "pz_p2z", "soft_label_qoc"]))
    return [row for row in rows if str(row.get("forensic_mode", "")) in requested]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_qoc_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t31_qoc_forensic_seed42.csv"), rows, T31_REQUIRED_FIELDS + ["forensic_mode", "assignment_mode", "assignment_hash", "assignment_overlap_with_other_modes", "num_codewords", "empty_codewords", "operator_topk", "operator_edges", "operator_row_sum_error", "identity_transfer_acc", "table_only_acc", "pz_only_acc", "pzp2_acc", "accuracy_delta_from_reference"])
    ensure_report(
        _arg(args, "report", "experiments/summaries/t31_qoc_forensic_notes.md"),
        [
            "# T31 QOC Forensic",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "forensic_mode", "accuracy", "identity_transfer_acc", "table_only_acc", "pz_only_acc", "pzp2_acc", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_qoc_forensic_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T31 QOC forensic rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--assignment-modes", nargs="+", default=["qoc_class_conditional_online_kmeans", "qoc_sft_ctc_assignment", "qoc_sft_bonsai_assignment", "qoc_hybrid_assignment"])
    parser.add_argument("--forensic-modes", nargs="+", default=["identity", "table_only", "pz_only", "pz_p2z", "soft_label_qoc"])
    parser.add_argument("--operator-topks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--students", nargs="+", default=["operator_sft_table_head"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--epochs", nargs="+", type=int, default=[60, 120])
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t31_qoc_forensic_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_qoc_forensic_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
