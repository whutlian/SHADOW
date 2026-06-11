from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t29_arxiv_cns_actual import build_arxiv_cns_server_command
from scripts.run_t29_arxiv_semantic_teacher import build_semantic_server_command
from scripts.run_t29_products_maintenance import build_products_server_command
from scripts.run_t29_reddit_bonsai import build_bonsai_server_command
from scripts.run_t29_reddit_control_audit import build_control_server_command
from scripts.run_t29_reddit_operator_match import build_omcp_server_command
from scripts.run_t29_reddit_pltc import build_pltc_server_command
from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t29_contract import ARXIV_A1, ARXIV_A2, ARXIV_A3, summarize_rows, validate_t29_row


STAGE_FIELDS = ["stage", "requirement_check", "requirement_status", "answer", "evidence", "blocked_reason", "next_command"]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row(check: str, status: str, answer: str, evidence: str = "", blocked_reason: str = "", next_command: str = "") -> dict[str, Any]:
    return {
        "stage": "t29",
        "requirement_check": check,
        "requirement_status": status,
        "answer": answer,
        "evidence": evidence,
        "blocked_reason": blocked_reason,
        "next_command": next_command,
    }


def _best(rows: list[dict[str, Any]], *, ratio: float | None = None, contains: str | None = None, track: str | None = None) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if row.get("accuracy") in {"", None}:
            continue
        if ratio is not None and abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) > 1e-12:
            continue
        if contains is not None and contains not in str(row.get("method", "")):
            continue
        if track is not None and str(row.get("promotion_track", "")) != track:
            continue
        candidates.append(row)
    return max(candidates, key=lambda row: fvalue(row.get("accuracy"))) if candidates else None


def _evidence(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return f"method={row.get('method')}, ratio={row.get('requested_full_node_ratio')}, acc={row.get('accuracy')}, macro={row.get('macro_f1')}, status={row.get('status')}"


def _improved(candidate: dict[str, Any] | None, reference: dict[str, Any] | None) -> bool:
    return bool(candidate and reference and fvalue(candidate.get("accuracy")) > fvalue(reference.get("accuracy")))


def build_stage_summary_rows(
    *,
    arxiv_cns_csv: str | Path = "experiments/tables/t29_arxiv_cns_actual_seed42.csv",
    semantic_csv: str | Path = "experiments/tables/t29_arxiv_semantic_teacher_seed42.csv",
    reddit_control_csv: str | Path = "experiments/tables/t29_reddit_control_audit_seed42.csv",
    omcp_csv: str | Path = "experiments/tables/t29_reddit_omcp_seed42.csv",
    pltc_csv: str | Path = "experiments/tables/t29_reddit_pltc_seed42.csv",
    bonsai_csv: str | Path = "experiments/tables/t29_reddit_bonsai_seed42.csv",
    products_csv: str | Path = "experiments/tables/t29_products_maintenance_seed42.csv",
) -> list[dict[str, Any]]:
    arxiv_cns = read_csv(arxiv_cns_csv)
    semantic = read_csv(semantic_csv)
    control = read_csv(reddit_control_csv)
    omcp = read_csv(omcp_csv)
    pltc = read_csv(pltc_csv)
    bonsai = read_csv(bonsai_csv)
    products = read_csv(products_csv)
    all_rows = arxiv_cns + semantic + control + omcp + pltc + bonsai + products

    actual_cns = [row for row in arxiv_cns if str(row.get("status", "")).startswith("completed_real")]
    raw_mlp = _best(arxiv_cns, contains="raw_mlp")
    best_safe_arxiv = _best(arxiv_cns, track="safe_mainline")
    best_semantic = _best(semantic)
    best_control_001 = _best(control, ratio=0.001)
    best_control_005 = _best(control, ratio=0.005)
    best_omcp_001 = _best(omcp, ratio=0.001)
    best_omcp_005 = _best(omcp, ratio=0.005)
    best_pltc = _best(pltc)
    best_omcp = _best(omcp)
    promoted = [row for row in all_rows if row.get("promotion_status") == "promoted" or _truthy(row.get("promotion_allowed", False))]
    unsafe = [row for row in promoted if not validate_t29_row(row)["valid"]]
    product_025 = _best(products, ratio=0.0025)
    product_050 = _best(products, ratio=0.005)
    products_ok = bool(product_025 and fvalue(product_025.get("accuracy")) >= 0.746 and product_050 and fvalue(product_050.get("accuracy")) >= 0.767)

    omcp_improved = _improved(best_omcp_001, best_control_001) or _improved(best_omcp_005, best_control_005)
    pltc_improved = _improved(best_pltc, best_omcp) or _improved(best_pltc, best_control_001) or _improved(best_pltc, best_control_005)

    return [
        _row(
            "arxiv_actual_cns_base_logits",
            "completed" if actual_cns else "blocked",
            f"actual_cns_rows={len(actual_cns)}",
            "; ".join(_evidence(row) for row in actual_cns[:3]),
            "" if actual_cns else "missing_base_logits_or_no_completed_real_rows",
            build_arxiv_cns_server_command(),
        ),
        _row(
            "arxiv_raw_mlp_cns_sanity_0p725",
            "completed" if raw_mlp and fvalue(raw_mlp.get("accuracy")) >= 0.725 else "blocked",
            f"raw_mlp_cns_passed={bool(raw_mlp and fvalue(raw_mlp.get('accuracy')) >= 0.725)}",
            _evidence(raw_mlp),
            "" if raw_mlp and fvalue(raw_mlp.get("accuracy")) >= 0.725 else "raw_mlp_cns_missing_or_below_0.725",
            build_arxiv_cns_server_command(),
        ),
        _row(
            "arxiv_safe_teacher_gates",
            "completed" if best_safe_arxiv and fvalue(best_safe_arxiv.get("accuracy")) >= ARXIV_A1 else "blocked",
            (
                f"A1={bool(best_safe_arxiv and fvalue(best_safe_arxiv.get('accuracy')) >= ARXIV_A1)}, "
                f"A2={bool(best_safe_arxiv and fvalue(best_safe_arxiv.get('accuracy')) >= ARXIV_A2)}, "
                f"A3={bool(best_safe_arxiv and fvalue(best_safe_arxiv.get('accuracy')) >= ARXIV_A3)}"
            ),
            _evidence(best_safe_arxiv),
            "" if best_safe_arxiv and fvalue(best_safe_arxiv.get("accuracy")) >= ARXIV_A1 else "safe_arxiv_teacher_below_A1_or_missing",
        ),
        _row(
            "arxiv_semantic_teacher_gates",
            "completed" if best_semantic and fvalue(best_semantic.get("accuracy")) >= 0.740 else "blocked",
            (
                f"semantic_0.740={bool(best_semantic and fvalue(best_semantic.get('accuracy')) >= 0.740)}, "
                f"semantic_0.755={bool(best_semantic and fvalue(best_semantic.get('accuracy')) >= 0.755)}"
            ),
            _evidence(best_semantic),
            "" if best_semantic and fvalue(best_semantic.get("accuracy")) >= 0.740 else "semantic_teacher_missing_or_below_0.740",
            build_semantic_server_command(),
        ),
        _row(
            "reddit_omcp_improves_0p10_or_0p50",
            "completed" if omcp_improved else "blocked",
            f"omcp_improved={omcp_improved}",
            f"omcp001={_evidence(best_omcp_001)}; ref001={_evidence(best_control_001)}; omcp005={_evidence(best_omcp_005)}; ref005={_evidence(best_control_005)}",
            "" if omcp_improved else "omcp_has_no_accuracy_or_no_improvement_yet",
            build_omcp_server_command(),
        ),
        _row(
            "reddit_pltc_improves_sota_chase",
            "completed" if pltc_improved else "blocked",
            f"pltc_improved={pltc_improved}",
            f"pltc={_evidence(best_pltc)}; omcp={_evidence(best_omcp)}",
            "" if pltc_improved else "pltc_has_no_accuracy_or_no_improvement_yet",
            build_pltc_server_command(),
        ),
        _row(
            "promoted_safe_rows",
            "completed" if any(row.get("promotion_track") == "safe_mainline" for row in promoted) else "blocked",
            f"safe_promoted={sum(1 for row in promoted if row.get('promotion_track') == 'safe_mainline')}",
            json.dumps(summarize_rows(all_rows), sort_keys=True),
            "no_safe_promoted_rows_yet" if not any(row.get("promotion_track") == "safe_mainline" for row in promoted) else "",
        ),
        _row(
            "promoted_sota_chase_rows",
            "completed" if any(row.get("promotion_track") == "sota_chase" for row in promoted) else "blocked",
            f"sota_promoted={sum(1 for row in promoted if row.get('promotion_track') == 'sota_chase')}",
            json.dumps(summarize_rows(all_rows), sort_keys=True),
            "no_sota_promoted_rows_yet" if not any(row.get("promotion_track") == "sota_chase" for row in promoted) else "",
        ),
        _row(
            "promoted_forbidden_guard",
            "completed" if not unsafe else "blocked",
            f"unsafe_promoted={len(unsafe)}",
            ",".join(str(row.get("method")) for row in unsafe),
            "" if not unsafe else "forbidden_promoted_rows_present",
        ),
        _row(
            "products_maintenance_regression",
            "completed" if products_ok else "blocked",
            f"products_ok={products_ok}",
            f"0.25={_evidence(product_025)}; 0.50={_evidence(product_050)}",
            "" if products_ok else "products_reference_missing_or_regressed",
            build_products_server_command(),
        ),
        _row(
            "next_server_commands",
            "completed",
            "Prepared all T29 server commands.",
            "arxiv_cns, semantic, control, omcp, pltc, bonsai, products",
            "",
            " ; ".join(
                [
                    build_arxiv_cns_server_command(),
                    build_semantic_server_command(),
                    build_control_server_command(),
                    build_omcp_server_command(),
                    build_pltc_server_command(),
                    build_bonsai_server_command(),
                    build_products_server_command(),
                ]
            ),
        ),
    ]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_stage_summary_rows(
        arxiv_cns_csv=args.arxiv_cns_csv,
        semantic_csv=args.semantic_csv,
        reddit_control_csv=args.reddit_control_csv,
        omcp_csv=args.omcp_csv,
        pltc_csv=args.pltc_csv,
        bonsai_csv=args.bonsai_csv,
        products_csv=args.products_csv,
    )
    csv_path = write_csv(args.csv, rows, STAGE_FIELDS)
    tables = {
        "arxiv_cns": read_csv(args.arxiv_cns_csv),
        "semantic": read_csv(args.semantic_csv),
        "reddit_control": read_csv(args.reddit_control_csv),
        "omcp": read_csv(args.omcp_csv),
        "pltc": read_csv(args.pltc_csv),
        "bonsai": read_csv(args.bonsai_csv),
        "products": read_csv(args.products_csv),
    }
    blocked = [row for row in rows if row.get("requirement_status") == "blocked"]
    ensure_report(
        args.report,
        [
            "# T29 Stage Summary",
            "",
            "## Files Changed",
            "- `shadow_hgc/sft/t29_contract.py`: strict schema, safe/SOTA guard, ratio accounting.",
            "- `shadow_hgc/sft/arxiv_actual_cns.py`: actual C&S grid that requires base logits.",
            "- `shadow_hgc/sft/semantic_arxiv_features.py`: local text/cache loader with no fabrication.",
            "- `shadow_hgc/sft/operator_match.py`: sparse row-stochastic OMCP fitting.",
            "- `shadow_hgc/sft/pseudo_label_transport.py`: PLTC selection and soft labels.",
            "- `shadow_hgc/sft/bonsai_sft_sketch.py`: LSH Bonsai sketch coverage.",
            "- `shadow_hgc/reddit/operator_students.py`: explicit weighted operator students.",
            "- `scripts/run_t29_*.py` and `tests/test_t29_*.py`: runners, summaries, tests.",
            "",
            "## Requirement Checks",
            *markdown_table(rows, ["requirement_check", "requirement_status", "answer", "blocked_reason"]),
            "",
            "## Tables",
            *markdown_table(tables["arxiv_cns"], ["method", "status", "accuracy", "macro_f1", "valid_acc", "failure_reason"]),
            "",
            *markdown_table(tables["semantic"], ["method", "status", "promotion_track", "uses_external_text_features", "failure_reason"]),
            "",
            *markdown_table(tables["reddit_control"], ["method", "requested_full_node_ratio", "actual_condensed_nodes", "accuracy", "macro_f1", "status"]),
            "",
            *markdown_table(tables["omcp"], ["method", "requested_full_node_ratio", "actual_condensed_nodes", "operator_edges", "operator_row_sum_error", "status", "accuracy", "failure_reason"]),
            "",
            *markdown_table(tables["pltc"], ["method", "requested_full_node_ratio", "actual_condensed_nodes", "promotion_track", "uses_teacher_logits", "status", "accuracy", "failure_reason"]),
            "",
            *markdown_table(tables["bonsai"], ["method", "requested_full_node_ratio", "actual_condensed_nodes", "uses_exact_pairwise", "status", "accuracy", "failure_reason"]),
            "",
            *markdown_table(tables["products"], ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status"]),
            "",
            "## Tests Run",
            "- T29 focused tests are recorded in final response after execution.",
            "- T28/T25-T27 regression tests are rerun before commit.",
            "",
            "## Forbidden Guard",
            "- Safe rows reject teacher logits, KD, dense P2, E-by-d materialization, full edge index on GPU, dense adjacency, exact pairwise, valid/test label inputs.",
            "- SOTA-chase rows may log teacher logits or external text but still reject valid/test label inputs and unsafe dense paths.",
            "",
            "## Blocked Work",
            *markdown_table(blocked, ["requirement_check", "blocked_reason", "next_command"]),
            "",
            f"- Stage CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T29 stage summary.")
    parser.add_argument("--arxiv-cns-csv", default="experiments/tables/t29_arxiv_cns_actual_seed42.csv")
    parser.add_argument("--semantic-csv", default="experiments/tables/t29_arxiv_semantic_teacher_seed42.csv")
    parser.add_argument("--reddit-control-csv", default="experiments/tables/t29_reddit_control_audit_seed42.csv")
    parser.add_argument("--omcp-csv", default="experiments/tables/t29_reddit_omcp_seed42.csv")
    parser.add_argument("--pltc-csv", default="experiments/tables/t29_reddit_pltc_seed42.csv")
    parser.add_argument("--bonsai-csv", default="experiments/tables/t29_reddit_bonsai_seed42.csv")
    parser.add_argument("--products-csv", default="experiments/tables/t29_products_maintenance_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t29_stage_summary_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_stage_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
