from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t30_arxiv_cns_actual import build_arxiv_cns_server_command
from scripts.run_t30_arxiv_qoc import build_arxiv_qoc_server_command
from scripts.run_t30_arxiv_semantic_teacher import build_semantic_server_command
from scripts.run_t30_products_maintenance import build_products_server_command
from scripts.run_t30_reddit_qoc import build_reddit_qoc_pltc_server_command, build_reddit_qoc_server_command
from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t30_contract import ARXIV_A1, ARXIV_A2, ARXIV_A3, summarize_guard, validate_t30_row


STAGE_FIELDS = ["stage", "requirement_check", "requirement_status", "answer", "evidence", "blocked_reason", "next_command"]


def _row(check: str, status: str, answer: str, evidence: str = "", blocked_reason: str = "", next_command: str = "") -> dict[str, Any]:
    return {
        "stage": "t30",
        "requirement_check": check,
        "requirement_status": status,
        "answer": answer,
        "evidence": evidence,
        "blocked_reason": blocked_reason,
        "next_command": next_command,
    }


def _best(rows: list[dict[str, Any]], *, ratio: float | None = None, dataset: str | None = None, contains: str | None = None) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("accuracy") in {"", None}:
            continue
        if ratio is not None and abs(fvalue(row.get("requested_full_node_ratio")) - ratio) > 1e-12:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if contains is not None and contains not in str(row.get("method", "")):
            continue
        candidates.append(row)
    return max(candidates, key=lambda item: fvalue(item.get("accuracy"))) if candidates else None


def _evidence(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return f"method={row.get('method')}, ratio={row.get('requested_full_node_ratio')}, acc={row.get('accuracy')}, macro={row.get('macro_f1')}, status={row.get('status')}"


def build_stage_summary_rows(
    *,
    arxiv_cns: list[dict[str, Any]] | None = None,
    semantic: list[dict[str, Any]] | None = None,
    reddit_qoc: list[dict[str, Any]] | None = None,
    arxiv_qoc: list[dict[str, Any]] | None = None,
    products: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    arxiv_cns = arxiv_cns or []
    semantic = semantic or []
    reddit_qoc = reddit_qoc or []
    arxiv_qoc = arxiv_qoc or []
    products = products or []
    all_rows = arxiv_cns + semantic + reddit_qoc + arxiv_qoc + products
    promoted = [row for row in all_rows if row.get("promotion_status") == "promoted"]
    unsafe = [row for row in promoted if not validate_t30_row(row)["valid"]]
    blocked = Counter(str(row.get("failure_reason", "")) for row in all_rows if str(row.get("failure_reason", "")))
    best_001 = _best(reddit_qoc, ratio=0.001, dataset="Reddit", contains="qoc")
    best_005 = _best(reddit_qoc, ratio=0.005, dataset="Reddit", contains="qoc")
    best_arxiv = _best(arxiv_cns + semantic, dataset="ogbn-arxiv")
    best_arxiv_acc = fvalue(best_arxiv.get("accuracy")) if best_arxiv else 0.0
    product_done = any(row.get("status") == "carried_forward_reference" for row in products)
    return [
        _row("promoted_safe_rows", "completed" if any(row.get("promotion_track") == "safe_main" for row in promoted) else "blocked", f"safe_promoted={sum(1 for row in promoted if row.get('promotion_track') == 'safe_main')}", json.dumps(summarize_guard(all_rows), sort_keys=True), "" if any(row.get("promotion_track") == "safe_main" for row in promoted) else "no_safe_promoted_rows_yet"),
        _row("promoted_sota_chase_rows", "completed" if any(row.get("promotion_track") == "sota_chase" for row in promoted) else "blocked", f"sota_promoted={sum(1 for row in promoted if row.get('promotion_track') == 'sota_chase')}", json.dumps(summarize_guard(all_rows), sort_keys=True), "" if any(row.get("promotion_track") == "sota_chase" for row in promoted) else "no_sota_promoted_rows_yet"),
        _row("blocked_rows_by_reason", "completed", json.dumps(dict(sorted(blocked.items())), sort_keys=True)),
        _row("best_reddit_0p10", "completed" if best_001 else "blocked", _evidence(best_001), _evidence(best_001), "" if best_001 else "no_real_qoc_transfer_eval_at_0p10", build_reddit_qoc_server_command()),
        _row("best_reddit_0p50", "completed" if best_005 else "blocked", _evidence(best_005), _evidence(best_005), "" if best_005 else "no_real_qoc_transfer_eval_at_0p50", build_reddit_qoc_server_command()),
        _row("best_arxiv_teacher", "completed" if best_arxiv else "blocked", _evidence(best_arxiv), _evidence(best_arxiv), "" if best_arxiv else "no_real_arxiv_teacher_metric", build_arxiv_cns_server_command()),
        _row("arxiv_A1_A2_A3_status", "completed" if best_arxiv_acc >= ARXIV_A1 else "blocked", f"A1={best_arxiv_acc >= ARXIV_A1}, A2={best_arxiv_acc >= ARXIV_A2}, A3={best_arxiv_acc >= ARXIV_A3}", _evidence(best_arxiv), "" if best_arxiv_acc >= ARXIV_A1 else "arxiv_teacher_below_A1_or_missing", build_arxiv_cns_server_command()),
        _row("products_maintenance_status", "completed" if product_done else "blocked", f"products_reference_rows={sum(1 for row in products if row.get('status') == 'carried_forward_reference')}", "", "" if product_done else "products_maintenance_missing", build_products_server_command()),
        _row("forbidden_guard_hits", "completed" if not unsafe else "blocked", f"unsafe_promoted={len(unsafe)}", ",".join(row.get("method", "") for row in unsafe), "" if not unsafe else "forbidden_promoted_rows_present"),
        _row("next_server_commands", "completed", "Prepared T30 server commands.", "", "", " ; ".join([build_reddit_qoc_server_command(), build_reddit_qoc_pltc_server_command(), build_arxiv_cns_server_command(), build_semantic_server_command(), build_arxiv_qoc_server_command(), build_products_server_command()])),
    ]


def write_outputs(args: argparse.Namespace) -> Path:
    arxiv_cns = read_csv(args.arxiv_cns_csv)
    semantic = read_csv(args.semantic_csv)
    reddit_qoc = read_csv(args.reddit_qoc_csv)
    reddit_qoc_multiseed = read_csv(args.reddit_qoc_multiseed_csv)
    arxiv_qoc = read_csv(args.arxiv_qoc_csv)
    products = read_csv(args.products_csv)
    rows = build_stage_summary_rows(arxiv_cns=arxiv_cns, semantic=semantic, reddit_qoc=reddit_qoc, arxiv_qoc=arxiv_qoc, products=products)
    csv_path = write_csv(args.csv, rows, STAGE_FIELDS)
    all_rows = arxiv_cns + semantic + reddit_qoc + arxiv_qoc + products
    ensure_report(
        args.report,
        [
            "# T30 Stage Summary",
            "",
            "## Files Changed",
            "- `shadow_hgc/sft/t30_contract.py`: strict T30 row schema and promotion guards.",
            "- `shadow_hgc/sft/codebook_assignment.py`: full-node codebook assignment helpers.",
            "- `shadow_hgc/sft/quotient_operator.py`: sparse destination-row codeword quotient operator.",
            "- `shadow_hgc/sft/qoc_condense.py` and `qoc_transfer_eval.py`: QOC table construction and table-head transfer path.",
            "- `shadow_hgc/sft/arxiv_logits.py` and `arxiv_cns_actual.py`: base logit cache loading and actual C&S wrapper.",
            "- `scripts/run_t30_*.py` and `tests/test_t30_*.py`: stage runners and checks.",
            "",
            "## Requirement Checks",
            *markdown_table(rows, ["requirement_check", "requirement_status", "answer", "blocked_reason"]),
            "",
            "## Reddit QOC Rows",
            *markdown_table(reddit_qoc, ["method", "requested_full_node_ratio", "num_codewords", "operator_topk", "transfer_eval_type", "accuracy", "macro_f1", "status", "failure_reason"]),
            "",
            "## Reddit QOC Multiseed Rows",
            *markdown_table(reddit_qoc_multiseed, ["method", "seed", "requested_full_node_ratio", "num_codewords", "operator_topk", "transfer_eval_type", "accuracy", "macro_f1", "status", "failure_reason"]),
            "",
            "## Arxiv Teacher Rows",
            *markdown_table(arxiv_cns + semantic, ["method", "status", "accuracy", "macro_f1", "valid_acc", "failure_reason", "notes"]),
            "",
            "## Arxiv QOC Rows",
            *markdown_table(arxiv_qoc, ["method", "requested_full_node_ratio", "status", "failure_reason", "notes"]),
            "",
            "## Products Maintenance Rows",
            *markdown_table(products, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason"]),
            "",
            "## Method Flags",
            "- Shadow-QOC-hard rows use `promotion_track=safe_main`, `uses_teacher_logits=False`, and sparse quotient operators.",
            "- Shadow-QOC-soft/PLTC rows use `promotion_track=sota_chase`, may set `uses_teacher_logits=True`, and still reject valid/test label inputs.",
            "- Semantic arxiv rows set `uses_external_text_features=True` and remain blocked if raw text/cache is missing.",
            "",
            "## Forbidden Guard",
            f"- Guard summary: `{json.dumps(summarize_guard(all_rows), sort_keys=True)}`",
            "",
            "## Remaining Blocked Work",
            *markdown_table([row for row in rows if row.get("requirement_status") == "blocked"], ["requirement_check", "blocked_reason", "next_command"]),
            "",
            f"- Stage CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T30 stage summary.")
    parser.add_argument("--arxiv-cns-csv", default="experiments/tables/t30_arxiv_cns_actual_seed42.csv")
    parser.add_argument("--semantic-csv", default="experiments/tables/t30_arxiv_semantic_teacher_seed42.csv")
    parser.add_argument("--reddit-qoc-csv", default="experiments/tables/t30_reddit_qoc_seed42.csv")
    parser.add_argument("--reddit-qoc-multiseed-csv", default="experiments/tables/t30_reddit_qoc_multiseed.csv")
    parser.add_argument("--arxiv-qoc-csv", default="experiments/tables/t30_arxiv_qoc_seed42.csv")
    parser.add_argument("--products-csv", default="experiments/tables/t30_products_maintenance.csv")
    parser.add_argument("--csv", default="experiments/tables/t30_stage_summary_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t30_stage_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
