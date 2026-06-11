from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.arxiv_semantic_cache_v2 import raw_text_map_is_readable, validate_semantic_memmap
from shadow_hgc.sft.t32_contract import T32_REQUIRED_FIELDS, apply_t32_promotion_guard, make_t32_row


ARXIV_NUM_NODES = 169_343


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_arxiv_semantic_server_command() -> str:
    return (
        "python scripts/run_t32_arxiv_semantic_sft.py --device cuda --semantic-device cuda "
        "--lm-models scibert specter2 --raw-text-map data/ogbn_arxiv/titleabs.tsv.gz "
        "--node-id-to-paper-id dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz "
        "--build-semantic-cache-if-missing --build-semantic-sft --enable-cns --run-long"
    )


def _blocked_row(args: argparse.Namespace, model: str, reason: str, *, notes: str = "") -> dict[str, Any]:
    return make_t32_row(
        dataset="ogbn-arxiv",
        method=f"arxiv_semantic_sft_{model}",
        seed=int(_arg(args, "seed", 42)),
        status="blocked",
        failure_reason=reason,
        promotion_track="sota_chase",
        promotion_status="not_promoted",
        uses_external_text_features=True,
        semantic_encoder=model,
        raw_text_map_path=str(_arg(args, "raw_text_map", "")),
        node_id_to_paper_id_path=str(_arg(args, "node_id_to_paper_id", "")),
        notes=notes,
        next_action=build_arxiv_semantic_server_command(),
    )


def build_semantic_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_text_map = str(_arg(args, "raw_text_map", ""))
    node_map = str(_arg(args, "node_id_to_paper_id", ""))
    precomputed = str(_arg(args, "use_precomputed_semantic_features", ""))
    models = [str(v) for v in _arg(args, "lm_models", ["scibert"])]
    raw_ok = bool(raw_text_map) and raw_text_map_is_readable(raw_text_map)
    node_map_ok = bool(node_map) and Path(node_map).exists()
    for model in models:
        if not precomputed and (not raw_ok or not node_map_ok):
            rows.append(_blocked_row(args, model, "raw_text_or_semantic_cache_missing"))
            continue
        if precomputed:
            dim = int(_arg(args, "semantic_dim", 0) or 0)
            if dim <= 0:
                rows.append(_blocked_row(args, model, "semantic_dim_missing", notes="precomputed memmap requires --semantic-dim"))
                continue
            diag = validate_semantic_memmap(
                embedding_path=precomputed,
                shape=(ARXIV_NUM_NODES, dim),
                num_nodes=ARXIV_NUM_NODES,
                dim=dim,
            )
            if diag.get("blocked"):
                rows.append(_blocked_row(args, model, str(diag.get("failure_reason", "semantic_cache_missing"))))
                continue
            row = make_t32_row(
                dataset="ogbn-arxiv",
                method=f"arxiv_semantic_sft_{model}",
                seed=int(_arg(args, "seed", 42)),
                status="blocked",
                failure_reason="semantic_sft_training_not_run",
                promotion_track="sota_chase",
                promotion_status="not_promoted",
                uses_external_text_features=True,
                semantic_encoder=model,
                semantic_cache_path=diag["semantic_cache_path"],
                semantic_dim=diag["semantic_dim"],
                semantic_cache_bytes=diag["semantic_cache_bytes"],
                raw_text_map_path=raw_text_map,
                node_id_to_paper_id_path=node_map,
                notes="aligned semantic cache validated; SFT/C&S training still required for a promotable row",
                next_action=build_arxiv_semantic_server_command(),
            )
            rows.append(apply_t32_promotion_guard(row))
            continue
        rows.append(_blocked_row(args, model, "semantic_cache_build_not_implemented"))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_semantic_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t32_arxiv_semantic_sft_seed42.csv"), rows, T32_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t32_arxiv_semantic_sft_notes.md"),
        [
            "# T32 Arxiv Semantic-SFT",
            "",
            *markdown_table(rows, ["method", "semantic_encoder", "semantic_cache_path", "semantic_dim", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_arxiv_semantic_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T32 semantic/raw-text SFT checks for ogbn-arxiv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lm-models", nargs="+", default=["scibert"])
    parser.add_argument("--raw-text-map", default="")
    parser.add_argument("--node-id-to-paper-id", default="")
    parser.add_argument("--use-precomputed-semantic-features", default="")
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--semantic-cache-dir", default="experiments/cache/t32_arxiv_semantic")
    parser.add_argument("--build-semantic-cache-if-missing", action="store_true")
    parser.add_argument("--build-semantic-sft", action="store_true")
    parser.add_argument("--teacher-heads", nargs="+", default=["semantic_mlp"])
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--temporal-decay-gammas", nargs="+", type=float, default=[0.01])
    parser.add_argument("--csv", default="experiments/tables/t32_arxiv_semantic_sft_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t32_arxiv_semantic_sft_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
