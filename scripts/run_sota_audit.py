from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.audit.config_checks import assert_or_mark_invalid, validate_variant_config
from shadow_hgc.audit.reporting import best_rows_by_dataset, markdown_table
from shadow_hgc.audit.schema_checks import coerce_list
from scripts.run_sota_common import write_csv


AUDIT_FIELDS = [
    "dataset",
    "variant",
    "status",
    "invalid_reasons",
    "warnings",
    "model_type",
    "target_type",
    "num_classes",
    "metapath_blocks",
    "path_lad_blocks",
    "compiled_blocks",
    "teacher_type",
    "teacher_train_acc",
    "teacher_val_acc",
    "teacher_predicted_class_count",
    "use_kd",
    "kd_lambda",
    "temperature",
    "predicted_class_count",
    "block_norm_stats_source",
    "ratio_mode",
    "requested_ratio",
    "requested_full_condensed_node_ratio",
    "total_condensed_node_ratio",
    "accuracy",
    "macro_f1",
    "source_log",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _json_list(value: Any) -> str:
    return json.dumps(coerce_list(value), ensure_ascii=True)


def _metapath_blocks(summary: dict, row: dict) -> list:
    blocks = coerce_list(summary.get("metapath_blocks"))
    if blocks:
        return blocks
    multi = summary.get("multiscale_metadata", {})
    if isinstance(multi, dict):
        blocks = coerce_list(multi.get("metapath_names"))
        if blocks:
            return blocks
        blocks = coerce_list(multi.get("blocks"))
        if blocks:
            return blocks
    return coerce_list(row.get("metapath_blocks"))


def _compiled_blocks(summary: dict) -> list:
    schema = summary.get("compiled_schema", {})
    if isinstance(schema, dict):
        blocks = schema.get("blocks", [])
        if isinstance(blocks, list):
            names = []
            for block in blocks:
                if isinstance(block, dict):
                    names.append(block.get("name", ""))
                else:
                    names.append(str(block))
            return [name for name in names if name]
    return coerce_list(summary.get("compiled_blocks"))


def _teacher(summary: dict) -> dict:
    teacher = summary.get("teacher")
    return teacher if isinstance(teacher, dict) else {}


def _audit_row(row: dict, *, source_log_root: Path) -> dict:
    source_log = row.get("source_log", "")
    source_path = Path(source_log)
    if source_log and not source_path.is_absolute():
        source_path = source_log_root / source_path
    summary = _read_json(source_path) if source_log else {}
    teacher = _teacher(summary)
    metapath_blocks = _metapath_blocks(summary, row)
    path_lad_blocks = coerce_list(summary.get("path_lad_blocks", row.get("path_lad_blocks")))
    compiled_blocks = _compiled_blocks(summary)
    audit = {
        "dataset": row.get("dataset", summary.get("dataset", "")),
        "variant": row.get("variant", summary.get("variant", "")),
        "status": row.get("status", summary.get("status", "completed")),
        "model_type": summary.get("model_type", row.get("model_type", "")),
        "target_type": summary.get("target_type", row.get("target_type", "")),
        "num_classes": summary.get("num_classes", summary.get("class_metadata", {}).get("num_classes_global", "")),
        "metapath_blocks": _json_list(metapath_blocks),
        "path_lad_blocks": _json_list(path_lad_blocks),
        "compiled_blocks": _json_list(compiled_blocks),
        "teacher_type": summary.get("teacher_type", row.get("teacher_type", teacher.get("type", ""))),
        "teacher_train_acc": summary.get("teacher_train_acc", teacher.get("train_acc", "")),
        "teacher_val_acc": summary.get("teacher_val_acc", teacher.get("val_acc", "")),
        "teacher_predicted_class_count": summary.get("teacher_predicted_class_count", teacher.get("predicted_class_count", "")),
        "use_kd": summary.get("use_kd", row.get("use_kd", teacher.get("use_kd", False))),
        "kd_lambda": summary.get("kd_lambda", summary.get("lambda_kd", summary.get("kd_weight", row.get("kd_lambda", "")))),
        "temperature": summary.get("temperature", summary.get("kd_temperature", row.get("temperature", ""))),
        "predicted_class_count": summary.get("predicted_class_count", row.get("predicted_class_count", "")),
        "block_norm_stats_source": summary.get("block_norm_stats_source", summary.get("compiled_block_stats_source", row.get("compiled_block_stats_source", ""))),
        "ratio_mode": summary.get("ratio_mode", row.get("ratio_mode", "")),
        "requested_ratio": row.get("requested_ratio", summary.get("requested_ratio", "")),
        "requested_full_condensed_node_ratio": row.get("requested_full_condensed_node_ratio", summary.get("requested_full_condensed_node_ratio", "")),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", row.get("total_condensed_node_ratio", "")),
        "accuracy": summary.get("accuracy", row.get("accuracy", "")),
        "macro_f1": summary.get("macro_f1", row.get("macro_f1", "")),
        "source_log": source_log,
        "feature_blocks": _json_list(["self", *metapath_blocks, *compiled_blocks]),
        "block_dims": summary.get("block_dims", {}),
        "path_lad_uses_train_labels_only": summary.get("path_lad_uses_train_labels_only", None),
        "path_lad_row_normalize": summary.get("path_lad_row_normalize", summary.get("path_lad_normalize", "")),
        "path_lad_leave_one_out_for_train": summary.get("path_lad_leave_one_out_for_train", None),
        "path_lad_hub_clip_thresholds": summary.get("path_lad_hub_clip_thresholds", ""),
        "prototype_mode": summary.get("prototype_mode", row.get("prototype_mode", "")),
        "ce_loss": summary.get("ce_loss", teacher.get("ce_loss", "")),
        "kd_loss": summary.get("kd_loss", teacher.get("kd_loss", "")),
    }
    if audit["status"] != "completed":
        audit["invalid_reasons"] = [f"status_not_completed:{audit['status']}"]
        audit["warnings"] = []
        audit["accuracy"] = None
        audit["macro_f1"] = None
        return audit
    checks = validate_variant_config(audit, {}, summary)
    return assert_or_mark_invalid(audit, checks)


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    best = best_rows_by_dataset(rows)
    invalid = [row for row in rows if row.get("status") != "completed"]
    lines = [
        "# SOTA Alignment Audit Seed 42",
        "",
        "This is a read-only audit over historical and clean SOTA JSON/CSV artifacts. Invalid rows keep their historical logs but are excluded from best-row summaries.",
        "",
        "## Valid Best Rows",
    ]
    lines.extend(markdown_table(list(best.values()), ["dataset", "variant", "accuracy", "macro_f1", "model_type", "total_condensed_node_ratio"]))
    lines.extend(["", "## Invalid / Failed Rows"])
    lines.extend(markdown_table(invalid, ["dataset", "variant", "status", "invalid_reasons", "source_log"]))
    lines.extend(["", "## Gate Notes", "", "- SeHGNN/meta-path rows require actual `model_type=sehgnn_lite`, non-empty blocks, dims, feature block list, and block norm source.", "- KD rows require teacher train/val quality, predicted class count, temperature/lambda, and separate CE/KD losses.", "- Path-LAD rows require train-label-only, row normalization, leave-one-out, and hub clipping diagnostics.", "", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SOTA rows and mark invalid configurations.")
    parser.add_argument("--inputs", nargs="*", default=[
        "experiments/tables/sota_small_seed42.csv",
        "experiments/tables/sota_medium_seed42.csv",
        "experiments/tables/sota_diagnostics_seed42.csv",
        "experiments/tables/sota_clean_small_seed42.csv",
        "experiments/tables/medium_no_diffusion_refine_seed42.csv",
    ])
    parser.add_argument("--output", default="experiments/tables/sota_audit_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/sota_audit_seed42.md")
    args = parser.parse_args()
    root = Path.cwd()
    rows: list[dict] = []
    for input_path in args.inputs:
        for row in _read_csv(Path(input_path)):
            rows.append(_audit_row(row, source_log_root=root))
    output = Path(args.output)
    write_csv(output, rows, AUDIT_FIELDS)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()

