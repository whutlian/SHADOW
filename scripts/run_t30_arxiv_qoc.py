from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t30_contract import ARXIV_A1, T30_REQUIRED_FIELDS, fvalue, make_t30_row, ratio_budget


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_arxiv_qoc_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_arxiv_qoc.py --device cuda --ratios 0.0025 0.005 "
        "--teacher-cache experiments/cache/arxiv_best_teacher_logits.pt "
        "--assignment-modes qoc_class_conditional_online_kmeans qoc_hybrid_assignment qoc_pltc_confidence_balanced "
        "--operator-topks 8 16 32 --students operator_sft_table_head --hidden-dims 256 512 "
        f"--epochs 120 200 --seed {int(seed)} --run-long"
    )


def _teacher_accuracy(path: str | Path) -> float | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    if target.suffix == ".json":
        meta = json.loads(target.read_text(encoding="utf-8"))
        return fvalue(meta.get("accuracy", meta.get("test_acc", "")), -1.0)
    return None


def build_arxiv_qoc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    ratios = [float(v) for v in _arg(args, "ratios", [0.0025, 0.005])]
    teacher_acc = _teacher_accuracy(_arg(args, "teacher_cache", ""))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("ogbn-arxiv", ratio)
        if teacher_acc is None or teacher_acc < ARXIV_A1:
            rows.append(
                make_t30_row(
                    dataset="ogbn-arxiv",
                    method="arxiv_qoc_hard_0p50" if ratio >= 0.005 else "arxiv_qoc_hard_0p25",
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    num_codewords=budget,
                    status="blocked",
                    promotion_track="safe_main",
                    failure_reason="teacher_gate_not_passed",
                    notes="Arxiv QOC is blocked until a real safe teacher reaches A1 >= 0.715.",
                    next_action=build_arxiv_qoc_server_command(seed),
                    transfer_eval_type="not_run_teacher_gate_blocked",
                )
            )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_qoc_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t30_arxiv_qoc_seed42.csv"), rows, T30_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t30_arxiv_qoc_notes.md"),
        [
            "# T30 Arxiv QOC",
            "",
            "- Arxiv QOC is not run before a real A1 teacher gate passes.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "num_codewords", "status", "failure_reason", "notes"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_arxiv_qoc_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T30 arxiv QOC gated by real teacher quality.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0025, 0.005])
    parser.add_argument("--teacher-cache", default="")
    parser.add_argument("--assignment-modes", nargs="+", default=["qoc_class_conditional_online_kmeans", "qoc_hybrid_assignment"])
    parser.add_argument("--operator-topks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--students", nargs="+", default=["operator_sft_table_head"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[120, 200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t30_arxiv_qoc_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t30_arxiv_qoc_notes.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
