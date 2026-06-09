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


def _coverage(row: dict[str, Any], dataset: str) -> float:
    safe = t21_safe_baseline(dataset)
    pred = _float(row, "predicted_class_count", default=safe["predicted_class_min"])
    return min(1.0, pred / max(1.0, float(safe["predicted_class_min"])))


def convert_rows(t2_rows: list[dict[str, str]], datasets: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in t2_rows:
        dataset = row.get("dataset", "")
        if dataset not in datasets:
            continue
        valid_acc = _float(row, "valid_acc")
        valid_f1 = _float(row, "valid_macro_f1")
        coverage = _coverage(row, dataset)
        safe = t21_safe_baseline(dataset)
        acc = _float(row, "accuracy", default=0.0)
        macro = _float(row, "macro_f1", default=0.0)
        rows.append(
            {
                "dataset": dataset,
                "row_kind": row.get("row_kind", ""),
                "model_type": row.get("model_type", ""),
                "selection_protocol": "single_valid_with_acc_macro_coverage",
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
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2.1 robust block selection audit table from T2 no-logits rows.")
    parser.add_argument("--input", default="experiments/tables/t2_sft_safe_block_selection_seed42.csv")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb", "ogbn-arxiv"])
    parser.add_argument("--output", default="experiments/tables/t21_sft_block_selection_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t21_sft_block_selection_summary.md")
    args = parser.parse_args()
    rows = convert_rows(read_csv(args.input), set(args.datasets))
    output = write_csv(args.output, rows, T21_FULLGRAPH_FIELDS)
    final = [row for row in rows if row.get("row_kind") == "final"]
    lines = [
        "# T2.1 Robust Block Selection Audit",
        "",
        "Rows reuse the T2 no-logits candidate outcomes and add the T2.1 coverage-aware selection score. No row uses logits, KD, dense P2, bounded edges, or E x d materialization.",
        "",
        *markdown_table(final, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "selection_score", "selected_blocks", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
