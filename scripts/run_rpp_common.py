from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def fmt(value, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_relation(summary: dict, key: str):
    values = [
        float(diag[key])
        for diag in summary.get("diagnostics", {}).values()
        if isinstance(diag, dict) and key in diag and diag[key] not in (None, "")
    ]
    return "" if not values else sum(values) / len(values)


def gates(summary: dict, key: str) -> str:
    diagnostics = summary.get("diagnostics", {})
    payload = diagnostics.get(key, summary.get(f"{key[:-1]}_values", {}))
    return json.dumps(payload or {}, sort_keys=True)


def base_row(path: Path, *, dataset: str, variant: str, summary: dict) -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": summary.get("seed", ""),
        "ratio": summary.get("ratio", summary.get("requested_target_ratio", "")),
        "ratio_percent": "" if summary.get("ratio", summary.get("requested_target_ratio", "")) in ("", None) else float(summary.get("ratio", summary.get("requested_target_ratio"))) * 100.0,
        "ratio_mode": summary.get("ratio_mode", ""),
        "model_type": summary.get("model_type", summary.get("model", "")),
        "feature_mode": summary.get("feature_mode", ""),
        "loss_type": summary.get("loss_type", summary.get("ablation", {}).get("loss_type", "")),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "prototype_train_acc": summary.get("prototype_train_acc", ""),
        "prototype_train_loss_start": summary.get("prototype_train_loss_start", summary.get("train_loss_start", "")),
        "prototype_train_loss_end": summary.get("prototype_train_loss_end", summary.get("train_loss_end", "")),
        "num_optimizer_steps": summary.get("num_optimizer_steps", ""),
        "final_logits_activation": summary.get("final_logits_activation", ""),
        "effective_target_ratio": summary.get("effective_target_ratio", ""),
        "shadow_node_ratio": summary.get("shadow_node_ratio", ""),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", ""),
        "total_condensed_edge_ratio": summary.get("total_condensed_edge_ratio", ""),
        "byte_size_compression": summary.get("byte_size_compression", ""),
        "effective_M_tau": summary.get("effective_M_tau", ""),
        "shadow_nodes_total": summary.get("shadow_nodes_total", ""),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "condensed_edges_total": summary.get("condensed_edges_total", ""),
        "skeleton_coverage": mean_relation(summary, "SkeletonMassCoverage"),
        "residual_energy": mean_relation(summary, "ResidualEnergy"),
        "shadow_recon_err": mean_relation(summary, "ShadowReconErr"),
        "relation_gates": gates(summary, "relation_gates"),
        "block_gates": gates(summary, "block_gates"),
        "condensation_time": summary.get("condensation_time", ""),
        "training_time": summary.get("training_time", ""),
        "inference_time": summary.get("inference_time", ""),
        "peak_cpu_ram": summary.get("peak_cpu_ram", summary.get("peak_cpu_ram_gb", "")),
        "peak_gpu_ram": summary.get("peak_gpu_ram", summary.get("peak_gpu_ram_gb", "")),
        "disk_bytes": summary.get("disk_bytes", ""),
        "status": summary.get("status", "completed"),
        "reason": summary.get("reason", ""),
        "source_log": str(path),
    }


def write_report(path: str | Path, *, title: str, rows: list[dict], csv_path: str | Path, previous_best: dict[str, float] | None = None) -> None:
    output = Path(path)
    completed = [row for row in rows if row.get("status") == "completed" and row.get("accuracy") not in ("", None)]
    failed = [row for row in rows if row.get("status") not in ("", "completed")]
    lines = [
        f"# {title}",
        "",
        "## Completed / OOM / Failed Rows",
        "",
        "| Dataset | Variant | Ratio | Model | Feature | Loss | Acc | Macro-F1 | Pred Classes | Status |",
        "|---|---|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        ratio = "" if row.get("ratio_percent") in ("", None) else f"{float(row['ratio_percent']):.1f}%"
        lines.append(
            f"| {row.get('dataset','')} | {row.get('variant','')} | {ratio} | {row.get('model_type','')} | "
            f"{row.get('feature_mode','')} | {row.get('loss_type','')} | {fmt(row.get('accuracy'))} | "
            f"{fmt(row.get('macro_f1'))} | {row.get('predicted_class_count','')} | {row.get('status','')} |"
        )
    lines.extend(["", "## Best Rows", ""])
    if completed:
        best_acc = max(completed, key=lambda row: float(row["accuracy"]))
        best_f1 = max(completed, key=lambda row: float(row["macro_f1"]) if row.get("macro_f1") not in ("", None) else -1.0)
        lines.append(f"- Best accuracy: `{fmt(best_acc['accuracy'])}` from `{best_acc['dataset']} / {best_acc['variant']}` at `{float(best_acc['ratio_percent']):.1f}%`.")
        lines.append(f"- Best macro-F1: `{fmt(best_f1['macro_f1'])}` from `{best_f1['dataset']} / {best_f1['variant']}` at `{float(best_f1['ratio_percent']):.1f}%`.")
    else:
        lines.append("- No completed accuracy rows.")
    if previous_best:
        lines.extend(["", "## Comparison To R+ Best", ""])
        for dataset, value in previous_best.items():
            subset = [row for row in completed if row.get("dataset") == dataset]
            if subset:
                best = max(subset, key=lambda row: float(row["accuracy"]))
                lines.append(f"- {dataset}: R++ best `{fmt(best['accuracy'])}` vs R+ `{fmt(value)}`.")
            else:
                lines.append(f"- {dataset}: no completed R++ row; R+ best `{fmt(value)}`.")
    lines.extend(
        [
            "",
            "## Compression And Resource Accounting",
            "",
            "| Dataset | Variant | Eff target ratio | Total node ratio | Edge ratio | Byte ratio | CPU RAM | GPU RAM | Disk bytes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in completed:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {fmt(row.get('effective_target_ratio'))} | "
            f"{fmt(row.get('total_condensed_node_ratio'))} | {fmt(row.get('total_condensed_edge_ratio'))} | "
            f"{fmt(row.get('byte_size_compression'))} | {row.get('peak_cpu_ram','')} | {row.get('peak_gpu_ram','')} | {row.get('disk_bytes','')} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| Dataset | Variant | Entropy | Relation gates | Block gates | Skel cov | Residual energy | Recon err |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in completed:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {fmt(row.get('prediction_entropy'))} | `{row.get('relation_gates','{}')}` | "
            f"`{row.get('block_gates','{}')}` | {fmt(row.get('skeleton_coverage'))} | {fmt(row.get('residual_energy'))} | {fmt(row.get('shadow_recon_err'))} |"
        )
    if failed:
        lines.extend(["", "## Failed Rows", ""])
        for row in failed:
            lines.append(f"- `{row.get('dataset')}/{row.get('variant')}` status `{row.get('status')}`: {row.get('reason','')}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- R++ rows are single seed 42 and should be interpreted as sprint diagnostics, not final multi-seed claims.",
            "- A row is considered scalable only when it reports completion rather than OOM/OOT.",
            "- Next recommendation is to keep R-1 defaults frozen and promote only opt-in R++ settings that improve accuracy without class collapse.",
            "",
            "## Files",
            "",
            f"- CSV: `{csv_path}`",
            f"- Report: `{output}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_rows(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows
