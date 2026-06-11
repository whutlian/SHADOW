from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.arxiv_semantic_cache_v3 import validate_semantic_cache_v3
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS, apply_t33_promotion_guard, make_t33_row


ARXIV_NUM_NODES = 169_343


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _raw_inputs_available(raw_text_map: str, node_id_to_paper_id: str) -> bool:
    return bool(raw_text_map) and bool(node_id_to_paper_id) and Path(raw_text_map).exists() and Path(node_id_to_paper_id).exists()


def _blocked_row(args: argparse.Namespace, method: str, encoder: str, reason: str, **fields: Any) -> dict[str, Any]:
    return make_t33_row(
        dataset="ogbn-arxiv",
        method=method,
        seed=int(_arg(args, "seed", 42)),
        status="blocked",
        failure_reason=reason,
        promotion_track="sota_chase",
        promotion_status="not_promoted",
        uses_external_text_features=True,
        semantic_encoder=encoder,
        **fields,
    )


def build_semantic_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_text_map = str(_arg(args, "raw_text_map", ""))
    node_map = str(_arg(args, "node_id_to_paper_id", ""))
    precomputed = str(_arg(args, "use_precomputed_semantic_features", ""))
    rows: list[dict[str, Any]] = []
    for encoder in [str(v) for v in _arg(args, "lm_models", ["scibert"])]:
        cache_method = f"arxiv_semantic_cache_{encoder}"
        if precomputed:
            diag = validate_semantic_cache_v3(metadata_path=precomputed, expected_num_nodes=ARXIV_NUM_NODES)
            if diag.get("blocked"):
                rows.append(_blocked_row(args, cache_method, encoder, str(diag.get("failure_reason", "semantic_cache_missing"))))
                continue
            rows.append(
                apply_t33_promotion_guard(
                    make_t33_row(
                        dataset="ogbn-arxiv",
                        method=cache_method,
                        seed=int(_arg(args, "seed", 42)),
                        status="completed_cache_validated",
                        failure_reason="semantic_sft_training_not_run",
                        promotion_track="semantic_teacher_diagnostic",
                        promotion_status="not_promoted",
                        uses_external_text_features=True,
                        semantic_encoder=diag.get("semantic_encoder", encoder),
                        semantic_cache_path=diag.get("semantic_cache_path", ""),
                        semantic_dim=diag.get("semantic_dim", ""),
                        cache_bytes=diag.get("semantic_cache_bytes", ""),
                        notes="Semantic cache validated; teacher training/C&S not run.",
                    )
                )
            )
            continue
        if not _raw_inputs_available(raw_text_map, node_map):
            rows.append(_blocked_row(args, cache_method, encoder, "raw_text_or_semantic_cache_missing"))
            continue
        rows.append(_blocked_row(args, cache_method, encoder, "semantic_cache_build_not_implemented"))
    for method in [
        "arxiv_semantic_mlp",
        "arxiv_semantic_sagn_lite",
        "arxiv_semantic_gamlp_lite",
        "arxiv_semantic_sagn_lite_cns",
        "arxiv_semantic_gamlp_lite_cns",
        "arxiv_semantic_time_cns",
    ]:
        rows.append(_blocked_row(args, method, "", "raw_text_or_semantic_cache_missing" if not precomputed else "semantic_teacher_training_not_run"))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_semantic_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t33_arxiv_semantic_sft.csv"), rows, T33_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t33_arxiv_semantic_sft.md"),
        ["# T33 Arxiv Semantic SFT", "", *markdown_table(rows, ["method", "semantic_encoder", "status", "failure_reason", "semantic_cache_path"]), "", f"- CSV: `{csv_path}`"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 arxiv semantic cache/SFT guard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--raw-text-map", default="")
    parser.add_argument("--node-id-to-paper-id", default="")
    parser.add_argument("--use-precomputed-semantic-features", default="")
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--lm-models", nargs="+", default=["scibert", "specter2", "e5"])
    parser.add_argument("--semantic-cache-dir", default="caches/arxiv_semantic")
    parser.add_argument("--build-semantic-sft", action="store_true")
    parser.add_argument("--teacher-heads", nargs="+", default=["semantic_mlp", "semantic_sagn_lite", "semantic_gamlp_lite"])
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_arxiv_semantic_sft.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_arxiv_semantic_sft.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
