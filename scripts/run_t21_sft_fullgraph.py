from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t21_common import T21_FULLGRAPH_FIELDS, markdown_table, read_csv, t21_safe_baseline, write_csv
from shadow_hgc.fullgraph.robust_block_selection import selection_score


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _convert_fullgraph(row: dict[str, str]) -> dict[str, Any]:
    dataset = row.get("dataset", "")
    safe = t21_safe_baseline(dataset)
    pred_min = max(1.0, float(safe["predicted_class_min"]))
    pred = _float(row, "predicted_class_count", default=0.0)
    coverage = min(1.0, pred / pred_min) if pred else 0.0
    valid_acc = _float(row, "valid_acc", default=0.0)
    valid_f1 = _float(row, "valid_macro_f1", default=0.0)
    acc = _float(row, "accuracy", default=0.0)
    macro = _float(row, "macro_f1", default=0.0)
    return {
        "dataset": dataset,
        "row_kind": "final",
        "model_type": row.get("model_type", ""),
        "selection_protocol": "acc_macro_coverage",
        "selection_score": selection_score(valid_acc=valid_acc, valid_macro_f1=valid_f1, class_coverage=coverage),
        "selected_blocks": row.get("selected_blocks", ""),
        "status": row.get("status", ""),
        "reason": row.get("reason", ""),
        "accuracy": row.get("accuracy", ""),
        "macro_f1": row.get("macro_f1", ""),
        "predicted_class_count": row.get("predicted_class_count", ""),
        "valid_acc": row.get("valid_acc", ""),
        "valid_macro_f1": row.get("valid_macro_f1", ""),
        "class_coverage": coverage,
        "safe_baseline": safe["variant"],
        "safe_baseline_acc": safe["accuracy"],
        "safe_baseline_macro_f1": safe["macro_f1"],
        "delta_acc_vs_safe": acc - float(safe["accuracy"]) if row.get("accuracy", "") != "" else "",
        "delta_macro_f1_vs_safe": macro - float(safe["macro_f1"]) if row.get("macro_f1", "") != "" else "",
        "full_edge_scans": row.get("full_edge_scans", ""),
        "edge_chunk_size": row.get("edge_chunk_size", ""),
        "dst_chunk_size": row.get("dst_chunk_size", ""),
        "feature_dim": row.get("block_dim", ""),
        "num_blocks": row.get("num_blocks", ""),
        "cache_bytes": row.get("cache_bytes", ""),
        "peak_cpu_ram_gb": row.get("peak_cpu_ram_gb", ""),
        "peak_gpu_ram_gb": row.get("peak_gpu_ram_gb", ""),
        "wall_time_s": row.get("wall_time_s", ""),
        "uses_memmap": row.get("uses_memmap", ""),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
        "uses_diffusion_legacy": False,
        "uses_full_graph_backprop": False,
        "uses_train_labels_only": row.get("uses_train_labels_only", ""),
        "source_log": row.get("source_log", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize T2.1 fullgraph table from no-logits rows.")
    parser.add_argument("--input", default="experiments/tables/t2_sft_fullgraph_seed42.csv")
    parser.add_argument("--products", default="experiments/tables/t21_products_full_execution_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t21_sft_fullgraph_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t21_sft_fullgraph_summary.md")
    args = parser.parse_args()
    rows = [_convert_fullgraph(row) for row in read_csv(args.input) if row.get("row_kind") in {"final", "resource_guard"}]
    products = read_csv(args.products)
    if products:
        prow = products[0]
        if prow.get("status") in {"promoted", "completed", "completed_non_regression", "preprop_completed", "blocked_full_execution_failed", "blocked_requires_explicit_full_run"}:
            safe = t21_safe_baseline("ogbn-products")
            rows = [row for row in rows if row.get("dataset") != "ogbn-products"]
            rows.append(
                {
                    "dataset": "ogbn-products",
                    "row_kind": "final",
                    "model_type": "sft_table_teacher_v2",
                    "selection_protocol": "full_products_execution",
                    "selection_score": "",
                    "selected_blocks": prow.get("selected_blocks", "[]"),
                    "status": prow.get("status", ""),
                    "reason": prow.get("reason", ""),
                    "accuracy": prow.get("accuracy", ""),
                    "macro_f1": prow.get("macro_f1", ""),
                    "predicted_class_count": prow.get("predicted_class_count", ""),
                    "safe_baseline": safe["variant"],
                    "safe_baseline_acc": safe["accuracy"],
                    "safe_baseline_macro_f1": safe["macro_f1"],
                    "full_edge_scans": prow.get("full_edge_scans", ""),
                    "edge_chunk_size": prow.get("edge_chunk_size", ""),
                    "dst_chunk_size": prow.get("dst_chunk_size", ""),
                    "feature_dim": prow.get("feature_dim", ""),
                    "num_blocks": "",
                    "cache_bytes": prow.get("total_cache_bytes", ""),
                    "peak_cpu_ram_gb": prow.get("peak_cpu_ram_gb", ""),
                    "peak_gpu_ram_gb": prow.get("peak_gpu_ram_gb", ""),
                    "wall_time_s": "",
                    "uses_memmap": True,
                    "uses_logits_as_input": False,
                    "uses_teacher_logits": False,
                    "uses_kd": False,
                    "uses_dense_p2": False,
                    "uses_bounded_edges": False,
                    "uses_e_by_d_materialization": False,
                    "uses_diffusion_legacy": False,
                    "uses_full_graph_backprop": False,
                    "uses_train_labels_only": False,
                    "source_log": "",
                }
            )
    output = write_csv(args.output, rows, T21_FULLGRAPH_FIELDS)
    lines = [
        "# T2.1 Fullgraph SFT Table",
        "",
        "This table contains no-logits fullgraph SFT rows plus the products execution row when available.",
        "",
        *markdown_table(rows, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "selected_blocks", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
