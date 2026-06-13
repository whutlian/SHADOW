from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.arxiv_semantic_stt import validate_precomputed_semantic_memmap
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, make_t34_row


ARXIV_NUM_NODES = 169_343


def _raw_inputs_available(raw_text_map: str, node_id_to_paper_id: str) -> bool:
    return bool(raw_text_map) and bool(node_id_to_paper_id) and Path(raw_text_map).exists() and Path(node_id_to_paper_id).exists()


def _cache_blocked_row(args: argparse.Namespace, encoder: str, reason: str, **fields: Any) -> dict[str, Any]:
    return make_t34_row(
        dataset="ogbn-arxiv",
        method=f"arxiv_semantic_cache_{encoder}",
        seed=int(args.seed),
        status="blocked",
        failure_reason=reason,
        promotion_track="semantic_sota",
        promotion_status="not_promoted",
        uses_external_text_features=True,
        semantic_encoder=encoder,
        next_action="provide raw title/abstract map or aligned semantic memmap",
        **fields,
    )


def build_cache_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    precomputed = str(args.precomputed_semantic_memmap or args.use_precomputed_semantic_features or "")
    for encoder in args.lm_models:
        if precomputed:
            diag = validate_precomputed_semantic_memmap(
                memmap_path=precomputed,
                semantic_node_order_checksum=args.semantic_node_order_checksum,
                expected_node_order_checksum=args.expected_node_order_checksum,
                num_nodes=ARXIV_NUM_NODES,
                semantic_dim=int(args.semantic_dim),
                semantic_dtype=str(args.semantic_dtype),
            )
            if diag.get("blocked"):
                rows.append(_cache_blocked_row(args, str(encoder), str(diag.get("failure_reason", "semantic_cache_missing")), semantic_cache_path=precomputed))
                continue
            rows.append(
                make_t34_row(
                    dataset="ogbn-arxiv",
                    method=f"arxiv_semantic_cache_{encoder}",
                    seed=int(args.seed),
                    status="completed_cache_validated",
                    failure_reason="semantic_teacher_training_not_run",
                    promotion_track="semantic_sota",
                    promotion_status="not_promoted",
                    uses_external_text_features=True,
                    semantic_encoder=str(encoder),
                    semantic_cache_path=diag.get("semantic_cache_path", ""),
                    semantic_cache_bytes=diag.get("semantic_cache_bytes", ""),
                    semantic_features_are_frozen=True,
                    lm_finetuned=False,
                    semantic_cache_memmap=True,
                    notes=json.dumps(diag, sort_keys=True),
                )
            )
            continue
        if not _raw_inputs_available(args.raw_text_map, args.node_id_to_paper_id):
            rows.append(_cache_blocked_row(args, str(encoder), "raw_text_or_semantic_cache_missing"))
        else:
            rows.append(_cache_blocked_row(args, str(encoder), "semantic_cache_build_not_implemented", semantic_cache_path=str(Path(args.semantic_cache_dir) / str(encoder))))
    return rows


def build_teacher_rows(args: argparse.Namespace, cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache_ready = any(row.get("status") == "completed_cache_validated" for row in cache_rows)
    rows: list[dict[str, Any]] = []
    for head in args.teacher_heads:
        rows.append(
            make_t34_row(
                dataset="ogbn-arxiv",
                method=f"arxiv_{head}",
                seed=int(args.seed),
                status="blocked",
                failure_reason="semantic_teacher_training_not_run" if cache_ready else "raw_text_or_semantic_cache_missing",
                promotion_track="semantic_sota",
                promotion_status="not_promoted",
                uses_external_text_features=True,
                semantic_encoder=",".join(args.lm_models),
                semantic_features_are_frozen=cache_ready,
                lm_finetuned=False,
                semantic_cache_memmap=cache_ready,
                teacher_gate_passed=False,
                next_action="run semantic feature build/teacher training after validated cache is available",
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    cache_rows = build_cache_rows(args)
    teacher_rows = build_teacher_rows(args, cache_rows)
    write_csv(args.cache_csv, cache_rows, T34_REQUIRED_FIELDS)
    write_csv(args.teacher_csv, teacher_rows, T34_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T34 Arxiv Semantic Teacher",
            "",
            "## Semantic Cache",
            "",
            *markdown_table(cache_rows, ["method", "semantic_encoder", "status", "failure_reason", "semantic_cache_path", "semantic_cache_bytes"]),
            "",
            "## Teacher Heads",
            "",
            *markdown_table(teacher_rows, ["method", "status", "failure_reason", "teacher_gate_passed"]),
        ],
    )
    return Path(args.teacher_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 arxiv semantic cache/teacher guard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--raw-text-map", default="")
    parser.add_argument("--node-id-to-paper-id", default="")
    parser.add_argument("--lm-models", nargs="+", default=["scibert", "specter2", "e5"])
    parser.add_argument("--semantic-cache-dir", default="caches/arxiv_semantic")
    parser.add_argument("--build-semantic-sft", action="store_true")
    parser.add_argument("--teacher-heads", nargs="+", default=["semantic_mlp", "semantic_sagn_lite", "semantic_gamlp_lite", "semantic_time_sagn", "semantic_sagn_lite_cns", "semantic_gamlp_lite_cns", "semantic_time_cns"])
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use-precomputed-semantic-features", default="")
    parser.add_argument("--precomputed-semantic-memmap", default="")
    parser.add_argument("--semantic-node-order-checksum", default="")
    parser.add_argument("--expected-node-order-checksum", default="")
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--semantic-dtype", default="fp16")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--cache-csv", default="experiments/tables/t34_arxiv_semantic_cache.csv")
    parser.add_argument("--teacher-csv", default="experiments/tables/t34_arxiv_semantic_teacher.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_arxiv_semantic_teacher.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
