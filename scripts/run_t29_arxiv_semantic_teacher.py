from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.semantic_arxiv_features import load_arxiv_raw_text_map, semantic_flags
from shadow_hgc.sft.t29_contract import T29_REQUIRED_FIELDS, make_t29_row


SEMANTIC_METHODS = {
    "sbert": "arxiv_text_sbert_sft",
    "scibert": "arxiv_text_scibert_sft",
    "specter": "arxiv_text_specter_sft",
    "e5": "arxiv_text_e5_sft",
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_semantic_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_arxiv_semantic_teacher.py --device cuda --semantic-device cuda "
        "--lm-models scibert specter e5 --semantic-cache-dir caches/arxiv_semantic "
        "--build-semantic-sft --teacher-heads mlp sagn_lite gamlp_lite --enable-cns "
        "--hidden-dims 512 768 --epochs 300 "
        f"--seed {int(seed)} --run-long"
    )


def build_semantic_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    models = [str(v) for v in _arg(args, "lm_models", ["specter"])]
    raw_text_path = _arg(args, "raw_text_path", "")
    precomputed = _arg(args, "use_precomputed_semantic_features", "")
    rows: list[dict[str, Any]] = []
    for model in models:
        method = SEMANTIC_METHODS.get(model, f"arxiv_text_{model}_sft")
        load = load_arxiv_raw_text_map(
            search_paths=[raw_text_path] if raw_text_path else [],
            precomputed_embedding_path=precomputed or None,
        )
        if not load.available:
            rows.append(
                make_t29_row(
                    dataset="ogbn-arxiv",
                    method=method,
                    seed=seed,
                    status="blocked",
                    promotion_status="not_promoted",
                    promotion_track="sota_chase",
                    failure_reason=load.failure_reason,
                    notes=load.actionable_message,
                    extra=semantic_flags(model_name=model, feature_dim=0, cache_bytes=0, raw_text_encoded=False, encode_time=0.0),
                )
            )
            continue
        rows.append(
            make_t29_row(
                dataset="ogbn-arxiv",
                method=method,
                seed=seed,
                status="server_ready_not_run",
                promotion_status="not_promoted",
                promotion_track="sota_chase",
                failure_reason="semantic_teacher_training_not_run",
                notes=f"Semantic input is available from {load.source_path}; run server command to encode/train.",
                extra=semantic_flags(model_name=model, feature_dim=0, cache_bytes=0, raw_text_encoded=not bool(precomputed), encode_time=0.0),
                source_table=load.source_path,
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_semantic_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_arxiv_semantic_teacher_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_arxiv_semantic_teacher_summary.md"),
        [
            "# T29 Arxiv Semantic Teacher",
            "",
            "- Semantic rows are SOTA-chase by default and never fabricate features.",
            "- Missing raw text or semantic memmap produces blocked rows with actionable messages.",
            "",
            *markdown_table(rows, ["method", "status", "promotion_track", "uses_external_text_features", "semantic_lm_model", "failure_reason", "notes"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_semantic_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 arxiv semantic/text SFT teacher declarations.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--lm-models", nargs="+", default=["scibert", "specter", "e5"])
    parser.add_argument("--semantic-cache-dir", default="caches/arxiv_semantic")
    parser.add_argument("--raw-text-path", default="")
    parser.add_argument("--use-precomputed-semantic-features", default="")
    parser.add_argument("--build-semantic-sft", action="store_true")
    parser.add_argument("--teacher-heads", nargs="+", default=["mlp", "sagn_lite", "gamlp_lite"])
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_arxiv_semantic_teacher_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_arxiv_semantic_teacher_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
