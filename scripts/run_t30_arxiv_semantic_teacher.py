from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.semantic_arxiv_features import load_arxiv_raw_text_map, semantic_flags
from shadow_hgc.sft.t30_contract import T30_REQUIRED_FIELDS, make_t30_row


SEMANTIC_METHODS = {
    "scibert": "arxiv_text_scibert_sft",
    "specter2": "arxiv_text_specter2_sft",
    "e5": "arxiv_text_e5_sft",
    "instructor": "arxiv_text_instructor_sft",
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_semantic_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_arxiv_semantic_teacher.py --device cuda --semantic-device cuda "
        "--raw-text-map /path/to/ogbn_arxiv_titleabs.tsv --semantic-cache-dir caches/arxiv_semantic "
        "--lm-models scibert specter2 e5 --build-semantic-sft --teacher-heads mlp sagn_lite gamlp_lite "
        f"--enable-cns --hidden-dims 512 768 --epochs 300 --seed {int(seed)} --run-long"
    )


def build_semantic_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    models = [str(v) for v in _arg(args, "lm_models", ["scibert"])]
    raw_text = _arg(args, "raw_text_map", "")
    precomputed = _arg(args, "use_precomputed_semantic_features", "")
    rows: list[dict[str, Any]] = []
    for model in models:
        load = load_arxiv_raw_text_map(search_paths=[raw_text] if raw_text else [], precomputed_embedding_path=precomputed or None)
        method = SEMANTIC_METHODS.get(model, f"arxiv_text_{model}_sft")
        flags = semantic_flags(model_name=model, feature_dim=0, cache_bytes=0, raw_text_encoded=load.available and not bool(precomputed), encode_time=0.0)
        if not load.available:
            rows.append(
                make_t30_row(
                    dataset="ogbn-arxiv",
                    method=method,
                    seed=seed,
                    status="blocked",
                    promotion_track="sota_chase",
                    failure_reason="raw_text_missing",
                    notes="Provide --raw-text-map PATH or --use-precomputed-semantic-features PATH; no text is fabricated.",
                    next_action=build_semantic_server_command(seed),
                    transfer_eval_type="teacher_eval",
                    extra=flags,
                )
            )
            continue
        rows.append(
            make_t30_row(
                dataset="ogbn-arxiv",
                method=method,
                seed=seed,
                status="blocked",
                promotion_track="sota_chase",
                failure_reason="semantic_teacher_metrics_missing",
                notes=f"Semantic input is available from {load.source_path}; train the semantic SFT teacher before promotion.",
                next_action=build_semantic_server_command(seed),
                transfer_eval_type="teacher_eval",
                source_table=load.source_path,
                extra=flags,
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_semantic_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t30_arxiv_semantic_teacher_seed42.csv"), rows, T30_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t30_arxiv_semantic_teacher_notes.md"),
        [
            "# T30 Arxiv Semantic Teacher",
            "",
            "- Raw text or a precomputed semantic memmap is required.",
            "- Rows remain blocked until real teacher metrics exist.",
            "",
            *markdown_table(rows, ["method", "status", "promotion_track", "uses_external_text_features", "semantic_lm_model", "accuracy", "failure_reason", "notes"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_semantic_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T30 arxiv semantic/raw-text teacher.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--raw-text-map", default="")
    parser.add_argument("--semantic-cache-dir", default="caches/arxiv_semantic")
    parser.add_argument("--lm-models", nargs="+", default=["scibert", "specter2", "e5"])
    parser.add_argument("--use-precomputed-semantic-features", default="")
    parser.add_argument("--build-semantic-sft", action="store_true")
    parser.add_argument("--teacher-heads", nargs="+", default=["mlp", "sagn_lite", "gamlp_lite"])
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t30_arxiv_semantic_teacher_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t30_arxiv_semantic_teacher_notes.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
