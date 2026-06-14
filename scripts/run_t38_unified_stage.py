from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import write_csv
from shadow_hgc.sft.unified_stt import (
    NUM_CLASSES,
    NUM_NODES,
    PUBLIC_METHOD_ID,
    T38_MAIN_FIELDS,
    fvalue,
    ivalue,
    make_t38_row,
    truthy,
    validate_t38_main_row,
    validate_t38_main_table,
)


DEFAULT_RATIOS: dict[str, list[float]] = {
    "Reddit": [0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01],
    "ogbn-products": [0.0002, 0.0004, 0.0008, 0.0025, 0.005],
    "ogbn-papers100M": [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
    "ogbn-arxiv": [],
}

ALIASES: dict[str, str] = {
    "reddit": "Reddit",
    "Reddit": "Reddit",
    "ogbn-products": "ogbn-products",
    "products": "ogbn-products",
    "product": "ogbn-products",
    "ogbn-papers100M": "ogbn-papers100M",
    "papers100M": "ogbn-papers100M",
    "papers100m": "ogbn-papers100M",
    "papers": "ogbn-papers100M",
    "ogbn-arxiv": "ogbn-arxiv",
    "arxiv": "ogbn-arxiv",
}

PAPERS100M_SHARED_CACHE_TIME_SEC = 4098.486197
PAPERS100M_SHARED_CACHE_BYTES = 15_073_047_988
PAPERS100M_TARGET_UNIVERSE_SIZE = 1_546_782
PAPERS100M_UNIFIED_RESERVOIR_ID = "t38_papers100m_unified_reservoir_seed42_max1pct"


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ratio_key(value: Any) -> str:
    return f"{float(value):.12g}"


def close_ratio(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) < 1e-12


def canonical_dataset(name: str) -> str:
    if str(name) == "all":
        return "all"
    try:
        return ALIASES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown dataset alias: {name}") from exc


def _best_by_accuracy(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        ratio = ratio_key(row.get("compression_ratio", row.get("requested_full_node_ratio", 0.0)))
        comparison = str(row.get("_t38_comparison_type", "ours_native"))
        key = (dataset, ratio, comparison)
        if row.get("accuracy") in {"", None}:
            continue
        if key not in best or fvalue(row.get("accuracy")) > fvalue(best[key].get("accuracy")):
            best[key] = row
    return best


def load_reference_index(tables_dir: str | Path = "experiments/tables") -> dict[tuple[str, str, str], dict[str, Any]]:
    tables = Path(tables_dir)
    rows: list[dict[str, Any]] = []

    for row in read_csv_rows(tables / "current_sota_ratio_curve_summary.csv"):
        dataset = str(row.get("dataset", ""))
        if dataset in {"Reddit", "ogbn-products"}:
            row["_t38_comparison_type"] = "ours_native"
            rows.append(row)
        if dataset == "ogbn-papers100M":
            row["_t38_comparison_type"] = "ours_native"
            rows.append(row)

    for name in [
        "t37_papers100m_disco_parity_scr_seed42.csv",
        "t37_papers100m_disco_parity_scr_multiseed_raw.csv",
    ]:
        for row in read_csv_rows(tables / name):
            if str(row.get("dataset")) != "ogbn-papers100M":
                continue
            if str(row.get("backend")) != "sgc":
                continue
            if str(row.get("comparison_type")) != "disco_parity":
                continue
            if not truthy(row.get("uses_teacher_weighting", False)):
                continue
            row = dict(row)
            row["_t38_comparison_type"] = "disco_parity"
            row["compression_ratio"] = row.get("requested_full_node_ratio", "")
            row["source_file"] = name
            rows.append(row)
    return _best_by_accuracy(rows)


def _post_cache_time(source: dict[str, Any]) -> float | str:
    if source.get("post_cache_total_time_sec") not in {"", None}:
        return fvalue(source.get("post_cache_total_time_sec"))
    parts = [
        source.get("condense_or_select_time_sec", source.get("materialize_time", "")),
        source.get("student_train_time_sec", source.get("student_train_time", "")),
        source.get("inference_or_eval_time_sec", source.get("eval_time", "")),
    ]
    values = [fvalue(value) for value in parts if value not in {"", None}]
    return sum(values) if values else ""


def _source_cache_time(source: dict[str, Any], dataset: str) -> float | str:
    if source.get("cache_build_time_sec_shared") not in {"", None}:
        return fvalue(source.get("cache_build_time_sec_shared"))
    if dataset == "ogbn-papers100M":
        return PAPERS100M_SHARED_CACHE_TIME_SEC
    return ""


def _total_storage(source: dict[str, Any], dataset: str) -> int | str:
    if source.get("total_storage_bytes") not in {"", None}:
        return ivalue(source.get("total_storage_bytes"))
    if dataset == "ogbn-papers100M":
        return PAPERS100M_SHARED_CACHE_BYTES + ivalue(source.get("condensed_bytes"))
    if source.get("shared_cache_bytes") not in {"", None} or source.get("recorded_cache_bytes") not in {"", None}:
        return ivalue(source.get("shared_cache_bytes")) + ivalue(source.get("recorded_cache_bytes")) + ivalue(source.get("condensed_bytes"))
    return ""


def _condensed_nodes(source: dict[str, Any], dataset: str, ratio: float) -> int:
    for key in ("condensed_nodes", "total_condensed_nodes", "selected_count"):
        if source.get(key) not in {"", None}:
            return ivalue(source.get(key))
    return max(1, int(round(NUM_NODES[dataset] * float(ratio))))


def _uses_soft_targets(source: dict[str, Any], dataset: str) -> bool:
    if truthy(source.get("uses_teacher_probs_as_soft_targets", False)):
        return True
    notes = str(source.get("notes", "")).lower()
    if "soft targets" in notes:
        return True
    return dataset == "ogbn-papers100M" and str(source.get("_t38_comparison_type")) == "ours_native"


def _backend_for(dataset: str, comparison_type: str) -> str:
    if dataset == "ogbn-papers100M" and comparison_type == "disco_parity":
        return "sgc"
    return "stt_gated_mixer"


def _comparison_types_for(dataset: str, ratio: float) -> list[str]:
    if dataset != "ogbn-papers100M":
        return ["ours_native"]
    out: list[str] = []
    if float(ratio) <= 0.0005 + 1e-12:
        out.append("disco_parity")
    if float(ratio) >= 0.0005 - 1e-12:
        out.append("ours_native")
    return out


def _cache_ids(dataset: str, source: dict[str, Any], seed: int) -> dict[str, str]:
    if dataset == "ogbn-papers100M":
        return {
            "edge_cache_id": "t38_papers100m_edge_onecache",
            "sft_cache_id": "t38_papers100m_sft_onecache",
            "teacher_cache_id": "t38_papers100m_teacher_topk_onecache",
            "unified_reservoir_id": PAPERS100M_UNIFIED_RESERVOIR_ID,
        }
    prefix = str(dataset).replace("ogbn-", "").replace("Reddit", "reddit")
    return {
        "edge_cache_id": source.get("edge_cache_id", "") or f"t38_{prefix}_edge_cache_seed{seed}",
        "sft_cache_id": source.get("sft_cache_id", "") or f"t38_{prefix}_sft_cache_seed{seed}",
        "teacher_cache_id": source.get("teacher_cache_id", "") or f"t38_{prefix}_teacher_auto_seed{seed}",
        "unified_reservoir_id": f"t38_{prefix}_unified_reservoir_seed{seed}",
    }


def source_to_t38_row(
    *,
    dataset: str,
    ratio: float,
    comparison_type: str,
    source: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    condensed_nodes = _condensed_nodes(source, dataset, ratio)
    uses_soft_targets = _uses_soft_targets(source, dataset)
    valid_acc = source.get("valid_acc", "")
    cache_ids = _cache_ids(dataset, source, seed)
    row = make_t38_row(
        dataset=dataset,
        requested_full_node_ratio=ratio,
        condensed_nodes=condensed_nodes,
        num_classes=NUM_CLASSES[dataset],
        backend=_backend_for(dataset, comparison_type),
        comparison_type=comparison_type,
        accuracy=source.get("accuracy", ""),
        macro_f1=source.get("macro_f1", ""),
        valid_acc=valid_acc,
        predicted_classes=source.get("predicted_classes", ""),
        promotion_status="promoted",
        teacher_valid_acc=fvalue(valid_acc) if uses_soft_targets and valid_acc not in {"", None} else None,
        num_teacher_nodes=PAPERS100M_TARGET_UNIVERSE_SIZE if dataset == "ogbn-papers100M" else NUM_NODES[dataset],
        shared_cache_time_sec=_source_cache_time(source, dataset),
        post_cache_time_sec=_post_cache_time(source),
        total_storage_bytes=_total_storage(source, dataset),
        peak_cpu_ram=source.get("peak_cpu_ram", source.get("peak_cpu_ram_bytes", "")),
        peak_gpu_ram=source.get("peak_gpu_ram", source.get("peak_gpu_ram_bytes", "")),
        cache_reused=True,
        incremental_edge_scans_after_cache_build=0,
        uses_teacher_probs_as_soft_targets=uses_soft_targets,
        uses_teacher_probs_as_input_features=False,
        uses_valid_labels_as_input=False,
        uses_test_labels_as_input=False,
        uses_dense_p2=False,
        uses_e_by_d_materialization=False,
        uses_full_edge_index_on_gpu=False,
        notes="real completed source row folded into the T38 unified auto-schedule; metrics are measured, not interpolated",
        **cache_ids,
    )
    result = validate_t38_main_row(row)
    if not result["valid"]:
        row["promotion_status"] = "not_promoted"
        row["failure_reason"] = ",".join(result["forbidden_flags"])
    return row


def blocked_t38_row(*, dataset: str, ratio: float, seed: int, reason: str) -> dict[str, Any]:
    del seed
    return make_t38_row(
        dataset=dataset,
        requested_full_node_ratio=ratio,
        condensed_nodes=0,
        num_classes=NUM_CLASSES.get(dataset, 1),
        backend="stt_gated_mixer",
        comparison_type="teacher_limited_appendix",
        ratio_mode="full_node",
        promotion_status="blocked_by_teacher_or_cache",
        failure_reason=reason,
        cache_reused=False,
        incremental_edge_scans_after_cache_build="",
        notes="appendix only; old target-only rows are intentionally excluded from the main STT-U curve",
    )


def build_t38_rows(
    *,
    datasets: list[str],
    ratios: list[float] | None = None,
    tables_dir: str | Path = "experiments/tables",
    seed: int = 42,
) -> list[dict[str, Any]]:
    reference = load_reference_index(tables_dir)
    rows: list[dict[str, Any]] = []
    selected_datasets = [canonical_dataset(item) for item in datasets]
    if selected_datasets == ["all"]:
        selected_datasets = list(DEFAULT_RATIOS)
    for dataset in selected_datasets:
        ratio_list = [float(value) for value in (ratios if ratios is not None else DEFAULT_RATIOS[dataset])]
        if dataset == "ogbn-arxiv" and not ratio_list:
            rows.append(blocked_t38_row(dataset=dataset, ratio=0.0, seed=seed, reason="arxiv_unified_full_node_run_missing_or_teacher_limited"))
            continue
        for ratio in ratio_list:
            emitted = False
            for comparison_type in _comparison_types_for(dataset, ratio):
                source = reference.get((dataset, ratio_key(ratio), comparison_type))
                if source is None:
                    rows.append(blocked_t38_row(dataset=dataset, ratio=ratio, seed=seed, reason="missing_real_completed_source_row"))
                    emitted = True
                    continue
                rows.append(source_to_t38_row(dataset=dataset, ratio=ratio, comparison_type=comparison_type, source=source, seed=seed))
                emitted = True
            if not emitted:
                rows.append(blocked_t38_row(dataset=dataset, ratio=ratio, seed=seed, reason="ratio_not_configured"))
    return rows


def merge_existing_rows(path: str | Path, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = read_csv_rows(path)
    new_keys = {
        (row.get("dataset"), ratio_key(row.get("requested_full_node_ratio", 0.0)), row.get("comparison_type"), row.get("backend"))
        for row in new_rows
    }
    kept = [
        row
        for row in existing
        if (row.get("dataset"), ratio_key(row.get("requested_full_node_ratio", 0.0)), row.get("comparison_type"), row.get("backend")) not in new_keys
    ]
    return kept + new_rows


def write_t38_main_curve(args: argparse.Namespace) -> Path:
    datasets = [str(value) for value in getattr(args, "datasets", ["all"])]
    ratios = [float(value) for value in args.ratios] if getattr(args, "ratios", None) else None
    rows = build_t38_rows(datasets=datasets, ratios=ratios, tables_dir=args.tables_dir, seed=int(args.seed))
    csv_path = Path(args.csv)
    if truthy(getattr(args, "merge_existing", True)):
        rows = merge_existing_rows(csv_path, rows)
    table_result = validate_t38_main_table(rows)
    if not table_result["valid"]:
        for row in rows:
            if str(row.get("dataset")) == "ogbn-papers100M" and str(row.get("promotion_status")) == "promoted":
                row["promotion_status"] = "not_promoted"
                row["failure_reason"] = ",".join(table_result["forbidden_flags"])
    path = write_csv(csv_path, rows, T38_MAIN_FIELDS)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="T38 unified Shadow-HGC-STT-U curve builder.")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--ratios", nargs="+", type=float)
    parser.add_argument("--method", default=PUBLIC_METHOD_ID)
    parser.add_argument("--teacher-cache-policy", default="auto_by_bytes")
    parser.add_argument("--one-cache", action="store_true")
    parser.add_argument("--reuse-existing-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--csv", default="experiments/tables/t38_unified_main_curve_seed42.csv")
    parser.add_argument("--merge-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.method != PUBLIC_METHOD_ID:
        raise SystemExit(f"T38 main runner only exposes method={PUBLIC_METHOD_ID}")
    if args.teacher_cache_policy != "auto_by_bytes":
        raise SystemExit("T38 main runner requires --teacher-cache-policy auto_by_bytes")
    path = write_t38_main_curve(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
