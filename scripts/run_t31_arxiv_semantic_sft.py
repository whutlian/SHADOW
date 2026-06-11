from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.semantic_sft_blocks import validate_semantic_cache_alignment
from shadow_hgc.sft.t31_contract import ARXIV_NUM_NODES, T31_REQUIRED_FIELDS, make_t31_row


def build_semantic_server_command() -> str:
    return (
        "python scripts/run_t31_arxiv_semantic_sft.py --device cuda --semantic-device cuda "
        "--raw-text-map data/ogbn_arxiv/titleabs.tsv.gz --node-id-to-paper-id "
        "data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz --lm-models scibert specter2 e5 "
        "--semantic-cache-dir caches/arxiv_semantic --build-semantic-sft --teacher-heads mlp semantic_sagn_lite "
        "semantic_gamlp_lite --enable-cns --hidden-dims 512 768 --epochs 300 --seed 42 --run-long"
    )


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _has_path(value: str | Path | None) -> bool:
    return bool(value) and Path(str(value)).exists()


def build_semantic_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    models = list(_arg(args, "lm_models", ["scibert"]))
    raw_text = str(_arg(args, "raw_text_map", "") or "")
    node_map = str(_arg(args, "node_id_to_paper_id", "") or "")
    precomputed = str(_arg(args, "use_precomputed_semantic_features", "") or "")
    rows: list[dict[str, Any]] = []
    resources_available = _has_path(precomputed) or (_has_path(raw_text) and _has_path(node_map))
    for model_name in models:
        if not resources_available:
            rows.append(
                make_t31_row(
                    dataset="ogbn-arxiv",
                    method=f"arxiv_semantic_sft_{model_name}",
                    seed=int(_arg(args, "seed", 42)),
                    status="blocked",
                    failure_reason="raw_text_or_semantic_cache_missing",
                    promotion_track="sota_chase",
                    promotion_status="not_promoted",
                    uses_external_text_features=True,
                    semantic_model_name=model_name,
                    raw_text_map_path=raw_text,
                    node_id_to_paper_id_path=node_map,
                    semantic_cache_path=precomputed,
                    next_action=build_semantic_server_command(),
                )
            )
            continue
        if precomputed:
            diag = validate_semantic_cache_alignment(
                embedding_path=precomputed,
                shape=(ARXIV_NUM_NODES, int(_arg(args, "semantic_feature_dim", 768))),
                num_nodes=ARXIV_NUM_NODES,
                matched_nodes=ARXIV_NUM_NODES,
                min_match_rate=0.95,
            )
            status = "blocked" if diag["blocked"] else "completed_cache_loaded"
            reason = diag["failure_reason"] if diag["blocked"] else "semantic_teacher_training_not_run_in_smoke"
        else:
            diag = {
                "semantic_cache_path": "",
                "semantic_cache_bytes": "",
                "semantic_feature_dim": "",
                "semantic_match_rate": "",
                "semantic_unmatched_nodes": "",
            }
            status = "blocked"
            reason = "semantic_cache_not_built"
        rows.append(
            make_t31_row(
                dataset="ogbn-arxiv",
                method=f"arxiv_semantic_sft_{model_name}",
                seed=int(_arg(args, "seed", 42)),
                status=status,
                failure_reason=reason,
                promotion_track="sota_chase",
                promotion_status="not_promoted",
                uses_external_text_features=True,
                uses_valid_labels_for_hyperparam_selection=True,
                semantic_model_name=model_name,
                raw_text_map_path=raw_text,
                node_id_to_paper_id_path=node_map,
                semantic_cache_path=diag["semantic_cache_path"],
                semantic_cache_bytes=diag["semantic_cache_bytes"],
                semantic_feature_dim=diag["semantic_feature_dim"],
                semantic_match_rate=diag["semantic_match_rate"],
                semantic_unmatched_nodes=diag["semantic_unmatched_nodes"],
                uses_cns_postprocess=bool(_arg(args, "enable_cns", False)),
                next_action=build_semantic_server_command(),
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_semantic_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t31_arxiv_semantic_sft_seed42.csv"), rows, T31_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t31_arxiv_semantic_notes.md"),
        [
            "# T31 Arxiv Semantic SFT",
            "",
            *markdown_table(rows, ["method", "status", "failure_reason", "semantic_model_name", "semantic_cache_path", "semantic_match_rate"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_semantic_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T31 arxiv semantic SFT.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--raw-text-map", default="")
    parser.add_argument("--node-id-to-paper-id", default="")
    parser.add_argument("--use-precomputed-semantic-features", default="")
    parser.add_argument("--semantic-cache-dir", default="caches/arxiv_semantic")
    parser.add_argument("--semantic-feature-dim", type=int, default=768)
    parser.add_argument("--lm-models", nargs="+", default=["scibert", "specter2", "e5"])
    parser.add_argument("--build-semantic-sft", action="store_true")
    parser.add_argument("--teacher-heads", nargs="+", default=["mlp", "semantic_sagn_lite", "semantic_gamlp_lite"])
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t31_arxiv_semantic_sft_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_arxiv_semantic_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
