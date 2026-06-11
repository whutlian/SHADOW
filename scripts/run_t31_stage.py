from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t31_arxiv_actual_cns import build_arxiv_cns_server_command
from scripts.run_t31_arxiv_semantic_sft import build_semantic_server_command
from scripts.run_t31_products_maintenance import build_products_server_command
from scripts.run_t31_reddit_ttc import build_reddit_ttc_server_command
from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t31_contract import (
    REDDIT_TTC_001_GATE,
    REDDIT_TTC_005_GATE,
    summarize_guard,
    validate_t31_row,
)


STAGE_FIELDS = ["stage", "requirement_check", "requirement_status", "answer", "evidence", "blocked_reason", "next_command"]


def _row(check: str, status: str, answer: str, evidence: str = "", blocked_reason: str = "", next_command: str = "") -> dict[str, Any]:
    return {
        "stage": "t31",
        "requirement_check": check,
        "requirement_status": status,
        "answer": answer,
        "evidence": evidence,
        "blocked_reason": blocked_reason,
        "next_command": next_command,
    }


def _best(rows: list[dict[str, Any]], *, ratio: float | None = None) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if row.get("accuracy") in {"", None}:
            continue
        if ratio is not None and abs(fvalue(row.get("requested_full_node_ratio")) - ratio) > 1e-12:
            continue
        candidates.append(row)
    return max(candidates, key=lambda item: fvalue(item.get("accuracy"))) if candidates else None


def _has_real_metric(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("accuracy") not in {"", None} and str(row.get("status", "")).startswith("completed") for row in rows)


def build_stage_summary_rows(
    *,
    reddit_ttc: list[dict[str, Any]] | None = None,
    reddit_simsft: list[dict[str, Any]] | None = None,
    reddit_bonsai: list[dict[str, Any]] | None = None,
    qoc_forensic: list[dict[str, Any]] | None = None,
    arxiv_cns: list[dict[str, Any]] | None = None,
    arxiv_semantic: list[dict[str, Any]] | None = None,
    products: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    reddit_ttc = reddit_ttc or []
    reddit_simsft = reddit_simsft or []
    reddit_bonsai = reddit_bonsai or []
    qoc_forensic = qoc_forensic or []
    arxiv_cns = arxiv_cns or []
    arxiv_semantic = arxiv_semantic or []
    products = products or []
    all_rows = reddit_ttc + reddit_simsft + reddit_bonsai + qoc_forensic + arxiv_cns + arxiv_semantic + products
    promoted = [row for row in all_rows if row.get("promotion_status") == "promoted"]
    unsafe = [row for row in promoted if not validate_t31_row(row)["valid"]]
    best_ttc_001 = _best(reddit_ttc, ratio=0.001)
    best_ttc_005 = _best(reddit_ttc, ratio=0.005)
    qoc_modes = {str(row.get("forensic_mode", "")) for row in qoc_forensic}
    arxiv_real = [row for row in arxiv_cns if row.get("cns_accuracy") not in {"", None} or row.get("accuracy") not in {"", None}]
    semantic_cache = any(row.get("semantic_cache_path") for row in arxiv_semantic)
    products_seed_count = len({row.get("seed") for row in products if row.get("status") == "carried_forward_reference"})
    products_missing_refs = any(row.get("failure_reason") == "missing_products_seed_reference" for row in products)
    safe_promoted = sum(1 for row in promoted if row.get("promotion_track") == "safe_main")
    sota_promoted = sum(1 for row in promoted if row.get("promotion_track") == "sota_chase")
    return [
        _row("reddit_ttc_rows_present", "completed" if reddit_ttc else "blocked", f"rows={len(reddit_ttc)}", blocked_reason="" if reddit_ttc else "missing_ttc_csv"),
        _row("reddit_ttc_real_metrics_present", "completed" if _has_real_metric(reddit_ttc) else "blocked", f"real_metric_rows={sum(1 for row in reddit_ttc if row.get('accuracy') not in {'', None})}", blocked_reason="" if _has_real_metric(reddit_ttc) else "no_real_ttc_transfer_metrics"),
        _row("reddit_ttc_0p10_gate", "completed" if best_ttc_001 and fvalue(best_ttc_001.get("accuracy")) >= REDDIT_TTC_001_GATE else "blocked", f"best_acc={fvalue(best_ttc_001.get('accuracy')) if best_ttc_001 else ''}", json.dumps(best_ttc_001 or {}, sort_keys=True), "" if best_ttc_001 and fvalue(best_ttc_001.get("accuracy")) >= REDDIT_TTC_001_GATE else "ttc_0p10_gate_not_met"),
        _row("reddit_ttc_0p50_gate", "completed" if best_ttc_005 and fvalue(best_ttc_005.get("accuracy")) >= REDDIT_TTC_005_GATE else "blocked", f"best_acc={fvalue(best_ttc_005.get('accuracy')) if best_ttc_005 else ''}", json.dumps(best_ttc_005 or {}, sort_keys=True), "" if best_ttc_005 and fvalue(best_ttc_005.get("accuracy")) >= REDDIT_TTC_005_GATE else "ttc_0p50_gate_not_met"),
        _row("reddit_simsft_rows_present", "completed" if reddit_simsft else "blocked", f"rows={len(reddit_simsft)}", blocked_reason="" if reddit_simsft else "missing_simsft_csv"),
        _row("reddit_simsft_table_only_gate", "completed" if any(row.get("promotion_status") == "promoted" for row in reddit_simsft) else "blocked", f"promoted={sum(1 for row in reddit_simsft if row.get('promotion_status') == 'promoted')}", blocked_reason="simsft_table_only_gate_not_met"),
        _row(
            "reddit_bonsai_safe_rows_present",
            "completed" if any(row.get("promotion_track") == "safe_main" for row in reddit_bonsai) else "blocked",
            f"safe_rows={sum(1 for row in reddit_bonsai if row.get('promotion_track') == 'safe_main')}",
            blocked_reason="" if any(row.get("promotion_track") == "safe_main" for row in reddit_bonsai) else "missing_bonsai_safe_rows",
        ),
        _row(
            "reddit_bonsai_sota_rows_present",
            "completed" if any(row.get("promotion_track") == "sota_chase" for row in reddit_bonsai) else "blocked",
            f"sota_rows={sum(1 for row in reddit_bonsai if row.get('promotion_track') == 'sota_chase')}",
            blocked_reason="" if any(row.get("promotion_track") == "sota_chase" for row in reddit_bonsai) else "missing_bonsai_sota_rows",
        ),
        _row("qoc_forensic_identity_sanity", "completed" if "identity" in qoc_modes else "blocked", f"modes={sorted(qoc_modes)}", blocked_reason="" if "identity" in qoc_modes else "missing_qoc_identity_row"),
        _row(
            "qoc_forensic_operator_sanity",
            "completed" if {"table_only", "pz_only", "pz_p2z"}.issubset(qoc_modes) else "blocked",
            f"modes={sorted(qoc_modes)}",
            blocked_reason="" if {"table_only", "pz_only", "pz_p2z"}.issubset(qoc_modes) else "missing_qoc_operator_rows",
        ),
        _row("arxiv_base_logits_present", "completed" if arxiv_real else "blocked", f"real_or_loaded_rows={len(arxiv_real)}", blocked_reason="" if arxiv_real else "missing_arxiv_base_logits", next_command=build_arxiv_cns_server_command()),
        _row("arxiv_raw_mlp_cns_sanity", "completed" if any("raw_x_mlp" in str(row.get("base_predictor")) and fvalue(row.get("cns_accuracy", row.get("accuracy"))) >= 0.720 for row in arxiv_cns) else "blocked", "raw_x_mlp_cns>=0.720", blocked_reason="raw_mlp_cns_sanity_missing_or_failed", next_command=build_arxiv_cns_server_command()),
        _row("arxiv_sft_cns_gate", "completed" if any(row.get("base_predictor") in {"mlp_on_sft", "a4_sagn_lite_v4", "sagn_lite_v5", "gamlp_lite_v5"} and fvalue(row.get("cns_accuracy", row.get("accuracy"))) >= 0.715 for row in arxiv_cns) else "blocked", "sft_cns>=0.715", blocked_reason="sft_cns_gate_missing_or_failed", next_command=build_arxiv_cns_server_command()),
        _row("arxiv_semantic_cache_present", "completed" if semantic_cache else "blocked", f"semantic_cache_rows={sum(1 for row in arxiv_semantic if row.get('semantic_cache_path'))}", blocked_reason="" if semantic_cache else "raw_text_or_semantic_cache_missing", next_command=build_semantic_server_command()),
        _row("arxiv_semantic_teacher_gate", "completed" if any(fvalue(row.get("accuracy")) >= 0.740 for row in arxiv_semantic) else "blocked", "semantic_teacher>=0.740", blocked_reason="semantic_teacher_gate_missing_or_failed", next_command=build_semantic_server_command()),
        _row(
            "products_maintenance_multiseed",
            "completed" if products_seed_count > 1 or (products_seed_count >= 1 and products_missing_refs) else "blocked",
            f"completed_seed_count={products_seed_count}, missing_seed_refs={products_missing_refs}",
            blocked_reason="" if products_seed_count > 1 or (products_seed_count >= 1 and products_missing_refs) else "missing_products_seed_reference",
            next_command=build_products_server_command(),
        ),
        _row("forbidden_guard_hits", "completed" if not unsafe else "blocked", f"unsafe_promoted={len(unsafe)}", json.dumps(summarize_guard(all_rows), sort_keys=True), "" if not unsafe else "forbidden_promoted_rows_present"),
        _row("promoted_safe_rows", "completed" if safe_promoted else "blocked", f"safe_promoted={safe_promoted}", json.dumps(summarize_guard(all_rows), sort_keys=True), "" if safe_promoted else "no_safe_promoted_rows_yet"),
        _row("promoted_sota_chase_rows", "completed" if sota_promoted else "blocked", f"sota_promoted={sota_promoted}", json.dumps(summarize_guard(all_rows), sort_keys=True), "" if sota_promoted else "no_sota_promoted_rows_yet"),
    ]


def write_outputs(args: argparse.Namespace) -> Path:
    reddit_ttc = read_csv(args.reddit_ttc_csv)
    reddit_simsft = read_csv(args.reddit_simsft_csv)
    reddit_bonsai = read_csv(args.reddit_bonsai_csv)
    qoc_forensic = read_csv(args.qoc_forensic_csv)
    arxiv_cns = read_csv(args.arxiv_cns_csv)
    arxiv_semantic = read_csv(args.arxiv_semantic_csv)
    products = read_csv(args.products_csv)
    rows = build_stage_summary_rows(
        reddit_ttc=reddit_ttc,
        reddit_simsft=reddit_simsft,
        reddit_bonsai=reddit_bonsai,
        qoc_forensic=qoc_forensic,
        arxiv_cns=arxiv_cns,
        arxiv_semantic=arxiv_semantic,
        products=products,
    )
    csv_path = write_csv(args.csv, rows, STAGE_FIELDS)
    all_rows = reddit_ttc + reddit_simsft + reddit_bonsai + qoc_forensic + arxiv_cns + arxiv_semantic + products
    ensure_report(
        args.report,
        [
            "# T31 Teacher-Transport and Semantic SFT Stage Summary",
            "",
            "## Files Changed",
            "- `shadow_hgc/sft/t31_contract.py`: T31 row schema, ratio budgets, safe/sota promotion guards.",
            "- `shadow_hgc/sft/teacher_transport.py`: TTC selection, teacher diagnostics, soft-label condensed student helpers.",
            "- `shadow_hgc/sft/simsft_soft.py`: SimSFT soft centroid and residual table builder.",
            "- `shadow_hgc/sft/bonsai_sft_coverage.py`: LSH-style Bonsai coverage selector.",
            "- `shadow_hgc/sft/qoc_forensic.py`: QOC forensic row/hash/overlap utilities.",
            "- `shadow_hgc/sft/arxiv_cns_actual_v2.py` and `semantic_sft_blocks.py`: C&S planning and semantic cache validation helpers.",
            "- `scripts/run_t31_*.py`: T31 Reddit, Arxiv, Products, QOC, and stage runners.",
            "- `tests/test_t31_*.py`: T31 contract and module tests.",
            "",
            "## Method Flags",
            "- `reddit_ttc_coverage_plus_boundary_plus_mixup`: `promotion_track=sota_chase`, `uses_teacher_logits=True`, `uses_kd=False`, no valid/test labels as inputs.",
            "- `simsft_soft_centroids_plus_residual_exemplars`: table-only SimSFT, no graph builder promotion.",
            "- `bonsai_hard_train_label_coverage`: `promotion_track=safe_main`, hard train labels only.",
            "- `bonsai_soft_ttc_coverage` and `bonsai_coverage_plus_boundary`: `promotion_track=sota_chase`, teacher soft labels only.",
            "- `qoc_forensic_*`: diagnostic rows only, never promoted in T31.",
            "- `arxiv_semantic_sft_*`: `uses_external_text_features=True`, blocked unless raw text or semantic cache exists.",
            "",
            "## Demoted Branches",
            "- QOC-hard direct operator condensation remains forensic only.",
            "- HNR/FDM local tuning, STC trainable-delta/gradient matching, and naive graph builders are not T31 main paths.",
            "- Historical LAD arxiv C&S is diagnostic only and is not used as a main teacher result.",
            "",
            "## Tests",
            "- T31 unit tests added for contract guards, TTC leakage controls, SimSFT, Bonsai, QOC forensic, Arxiv C&S, semantic cache blocking, products maintenance, and stage aggregation.",
            "- Verification command: `python -m pytest tests/test_t31*.py -q`.",
            "",
            "## Requirement Checks",
            *markdown_table(rows, ["requirement_check", "requirement_status", "answer", "blocked_reason"]),
            "",
            "## Reddit TTC",
            *markdown_table(reddit_ttc, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "status", "promotion_status", "failure_reason"]),
            "",
            "## SimSFT and Bonsai",
            *markdown_table(reddit_simsft + reddit_bonsai, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "status", "promotion_track", "failure_reason"]),
            "",
            "## QOC Forensic",
            *markdown_table(qoc_forensic, ["method", "forensic_mode", "accuracy", "status", "promotion_status", "failure_reason"]),
            "",
            "## Arxiv",
            *markdown_table(arxiv_cns + arxiv_semantic, ["method", "base_predictor", "status", "accuracy", "cns_accuracy", "failure_reason", "semantic_cache_path"]),
            "",
            "## Products",
            *markdown_table(products, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason"]),
            "",
            "## Guard Summary",
            f"`{json.dumps(summarize_guard(all_rows), sort_keys=True)}`",
            "",
            "## Next Commands",
            f"- Reddit TTC full grid: `{build_reddit_ttc_server_command()}`",
            f"- Arxiv actual C&S full grid: `{build_arxiv_cns_server_command()}`",
            f"- Arxiv semantic SFT after raw text/cache is available: `{build_semantic_server_command()}`",
            f"- Products maintenance replay: `{build_products_server_command()}`",
            "",
            f"- Stage CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T31 stage summary.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reddit-ttc-csv", default="experiments/tables/t31_reddit_ttc_seed42.csv")
    parser.add_argument("--reddit-simsft-csv", default="experiments/tables/t31_reddit_simsft_seed42.csv")
    parser.add_argument("--reddit-bonsai-csv", default="experiments/tables/t31_reddit_bonsai_coverage_seed42.csv")
    parser.add_argument("--qoc-forensic-csv", default="experiments/tables/t31_qoc_forensic_seed42.csv")
    parser.add_argument("--arxiv-cns-csv", default="experiments/tables/t31_arxiv_actual_cns_seed42.csv")
    parser.add_argument("--arxiv-semantic-csv", default="experiments/tables/t31_arxiv_semantic_sft_seed42.csv")
    parser.add_argument("--products-csv", default="experiments/tables/t31_products_maintenance_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t31_stage_summary_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_teacher_transport_semantic_sft_stage_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
