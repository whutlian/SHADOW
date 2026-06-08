from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SMALL_MAIN_FIELDS = [
    "generated_at",
    "git_commit",
    "config_hash",
    "run_id",
    "dataset",
    "method",
    "M_tau_requested",
    "M_tau_effective",
    "M_r_resolved_summary",
    "feature_dim",
    "projection_type",
    "loss_type",
    "model",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "condensation_time_mean",
    "training_time_mean",
    "inference_time_mean",
    "status",
]

MEDIUM_MAIN_FIELDS = [
    "generated_at",
    "git_commit",
    "config_hash",
    "run_id",
    "dataset",
    "method",
    "M_tau_requested",
    "M_tau_effective",
    "feature_dim",
    "projection_type",
    "loss_type",
    "model",
    "shadow_mode",
    "self_only",
    "accuracy",
    "macro_f1",
    "num_predicted_classes",
    "condensation_time",
    "training_time",
    "inference_time",
    "status",
]

MEDIUM_ABLATION_FIELDS = [
    "dataset",
    "ablation",
    "setting",
    "seed",
    "accuracy",
    "macro_f1",
    "skeleton_coverage_mean",
    "residual_energy_mean",
    "shadow_recon_err_mean",
    "num_predicted_classes",
    "prototype_train_acc",
    "status",
]

RATIO_MAIN_FIELDS = [
    "dataset",
    "method",
    "ratio",
    "ratio_base",
    "budget_mode",
    "feature_dim",
    "projection_type",
    "loss_type",
    "model",
    "requested_target_budget",
    "effective_target_prototypes",
    "M_requested",
    "M_effective",
    "num_train_target_nodes",
    "num_train_classes",
    "min_proto_per_class",
    "baseline_match_mode",
    "baseline_budget",
    "shadow_condensed_nodes_total",
    "shadow_nodes_total",
    "condensed_nodes_total",
    "condensed_edges_total",
    "effective_target_ratio",
    "condensed_node_ratio_to_train_target",
    "condensed_node_ratio_to_all_task_nodes",
    "accuracy",
    "accuracy_std",
    "macro_f1",
    "macro_f1_std",
    "predicted_classes",
    "prototype_train_acc",
    "shadow_recon_err_mean",
    "skeleton_coverage_mean",
    "residual_energy_mean",
    "condense_s",
    "train_s",
    "infer_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "status",
    "source_log",
]


RATIO_BUDGET_SUMMARY_FIELDS = [
    "dataset",
    "method",
    "budget_mode",
    "ratio",
    "ratio_base",
    "requested_target_budget",
    "effective_target_prototypes",
    "shadow_nodes_total",
    "condensed_nodes_total",
    "condensed_edges_total",
    "effective_target_ratio",
    "condensed_node_ratio_to_train_target",
    "condensed_node_ratio_to_all_task_nodes",
    "source_log",
]


def _load_json_logs(log_dir: str | Path) -> list[dict]:
    path = Path(log_dir)
    return [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]


def _load_json_logs_with_paths(log_dir: str | Path, *, recursive: bool = False) -> list[tuple[Path, dict]]:
    path = Path(log_dir)
    globber = path.rglob("*.json") if recursive else path.glob("*.json")
    rows = []
    for item in sorted(globber):
        rows.append((item, json.loads(item.read_text(encoding="utf-8"))))
    return rows


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _format_mean(values: list[float]) -> str:
    return "" if not values else f"{_mean(values):.6f}"


def _format_std(values: list[float]) -> str:
    return "" if not values else f"{_std(values):.6f}"


def _numeric_values(payloads: list[dict], key: str) -> list[float]:
    values = []
    for item in payloads:
        value = item.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def _sum_dict(payload: dict, key: str) -> int | str:
    value = payload.get(key)
    if not isinstance(value, dict):
        return ""
    return int(sum(int(v) for v in value.values()))


def _ratio_key(payload: dict) -> tuple:
    return (
        payload.get("dataset", ""),
        payload.get("method", payload.get("baseline", "Shadow-HGC-R-1")),
        payload.get("budget_mode", "count"),
        "" if payload.get("ratio") is None else str(payload.get("ratio")),
        payload.get("ratio_base", "train_target"),
        payload.get("requested_target_budget", payload.get("requested_M_tau", payload.get("M_tau", ""))),
        payload.get("baseline_match_mode", ""),
        payload.get("loss_type", payload.get("ablation", {}).get("loss_type", "")),
        payload.get("model", ""),
    )


def build_small_main_rows_from_logs(log_dir: str | Path) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for payload in _load_json_logs(log_dir):
        if payload.get("status", "completed") != "completed":
            continue
        if "accuracy" not in payload:
            continue
        key = (
            payload.get("dataset"),
            payload.get("method", payload.get("baseline", "Shadow-HGC-R-1")),
            payload.get("requested_M_tau", payload.get("M_tau")),
            payload.get("projection_type", ""),
            payload.get("loss_type", payload.get("ablation", {}).get("loss_type", "")),
            payload.get("model", ""),
        )
        grouped[key].append(payload)
    rows = []
    for key, payloads in grouped.items():
        first = payloads[0]
        accuracies = _numeric_values(payloads, "accuracy")
        macro_f1s = _numeric_values(payloads, "macro_f1")
        rows.append(
            {
                "generated_at": first.get("generated_at", ""),
                "git_commit": first.get("git_commit", ""),
                "config_hash": first.get("config_hash", ""),
                "run_id": first.get("run_id", ""),
                "dataset": first.get("dataset", ""),
                "method": first.get("method", first.get("baseline", "Shadow-HGC-R-1")),
                "M_tau_requested": str(first.get("requested_M_tau", first.get("M_tau", ""))),
                "M_tau_effective": str(first.get("effective_M_tau", "")),
                "M_r_resolved_summary": ";".join(f"{k}={v}" for k, v in first.get("resolved_M_r", {}).items()),
                "feature_dim": str(first.get("feature_dim", "")),
                "projection_type": first.get("projection_type", ""),
                "loss_type": first.get("loss_type", first.get("ablation", {}).get("loss_type", "")),
                "model": first.get("model", ""),
                "accuracy_mean": _format_mean(accuracies),
                "accuracy_std": _format_std(accuracies),
                "macro_f1_mean": _format_mean(macro_f1s),
                "macro_f1_std": _format_std(macro_f1s),
                "condensation_time_mean": f"{_mean([float(item.get('condensation_time', 0.0)) for item in payloads]):.6f}",
                "training_time_mean": f"{_mean([float(item.get('training_time', 0.0)) for item in payloads]):.6f}",
                "inference_time_mean": f"{_mean([float(item.get('inference_time', 0.0)) for item in payloads]):.6f}",
                "status": "completed",
            }
        )
    return sorted(rows, key=lambda row: (row["dataset"], row["method"], row["M_tau_requested"]))


def _mean_diag(payload: dict, key: str) -> str:
    values = [float(diag[key]) for diag in payload.get("diagnostics", {}).values() if key in diag]
    return "" if not values else f"{_mean(values):.6f}"


def _mean_payload_diag(payloads: list[dict], key: str) -> str:
    values = []
    for payload in payloads:
        for diag in payload.get("diagnostics", {}).values():
            if key in diag:
                values.append(float(diag[key]))
    return "" if not values else f"{_mean(values):.6f}"


def _first_nonempty(payload: dict, *keys: str):
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return ""


def build_ratio_main_rows_from_logs(log_dir: str | Path, *, recursive: bool = False) -> list[dict]:
    grouped: dict[tuple, list[tuple[Path, dict]]] = defaultdict(list)
    for path, payload in _load_json_logs_with_paths(log_dir, recursive=recursive):
        if path.name.endswith("_stress.json") or "_ks" in path.stem or "ablations" in path.parts:
            continue
        if payload.get("status", "completed") != "completed":
            grouped[_ratio_key(payload)].append((path, payload))
            continue
        if "accuracy" not in payload and payload.get("method") != "Full-WRL-GNN":
            continue
        grouped[_ratio_key(payload)].append((path, payload))

    rows = []
    for _, items in grouped.items():
        paths = [path for path, _ in items]
        payloads = [payload for _, payload in items]
        first = payloads[0]
        accuracies = _numeric_values(payloads, "accuracy")
        macro_f1s = _numeric_values(payloads, "macro_f1")
        requested = _first_nonempty(first, "requested_target_budget", "requested_M_tau", "M_tau")
        effective = _first_nonempty(first, "effective_target_prototypes", "effective_M_tau")
        condensed_nodes_total = _first_nonempty(first, "condensed_nodes_total")
        if condensed_nodes_total == "":
            condensed_nodes_total = _sum_dict(first, "condensed_nodes_by_type")
        condensed_edges_total = _first_nonempty(first, "condensed_edges_total")
        if condensed_edges_total == "":
            condensed_edges_total = _sum_dict(first, "condensed_edges_by_relation")
        rows.append(
            {
                "dataset": first.get("dataset", ""),
                "method": first.get("method", first.get("baseline", "Shadow-HGC-R-1")),
                "ratio": "" if first.get("ratio") is None else str(first.get("ratio")),
                "ratio_base": first.get("ratio_base", ""),
                "budget_mode": first.get("budget_mode", "count"),
                "feature_dim": first.get("feature_dim", ""),
                "projection_type": first.get("projection_type", ""),
                "loss_type": first.get("loss_type", first.get("ablation", {}).get("loss_type", "")),
                "model": first.get("model", ""),
                "requested_target_budget": requested,
                "effective_target_prototypes": effective,
                "M_requested": requested,
                "M_effective": effective,
                "num_train_target_nodes": first.get("num_train_target_nodes", ""),
                "num_train_classes": first.get("num_train_classes", first.get("num_classes_train", "")),
                "min_proto_per_class": first.get("min_proto_per_class", ""),
                "baseline_match_mode": first.get("baseline_match_mode", ""),
                "baseline_budget": first.get("baseline_budget", ""),
                "shadow_condensed_nodes_total": first.get("shadow_condensed_nodes_total", ""),
                "shadow_nodes_total": first.get("shadow_nodes_total", ""),
                "condensed_nodes_total": condensed_nodes_total,
                "condensed_edges_total": condensed_edges_total,
                "effective_target_ratio": first.get("effective_target_ratio", ""),
                "condensed_node_ratio_to_train_target": first.get("condensed_node_ratio_to_train_target", ""),
                "condensed_node_ratio_to_all_task_nodes": first.get("condensed_node_ratio_to_all_task_nodes", ""),
                "accuracy": _format_mean(accuracies),
                "accuracy_std": _format_std(accuracies),
                "macro_f1": _format_mean(macro_f1s),
                "macro_f1_std": _format_std(macro_f1s),
                "predicted_classes": first.get("predicted_classes", first.get("num_predicted_classes", "")),
                "prototype_train_acc": "" if first.get("prototype_train_acc") in (None, "") else f"{float(first['prototype_train_acc']):.6f}",
                "shadow_recon_err_mean": _mean_payload_diag(payloads, "ShadowReconErr"),
                "skeleton_coverage_mean": _mean_payload_diag(payloads, "SkeletonMassCoverage"),
                "residual_energy_mean": _mean_payload_diag(payloads, "ResidualEnergy"),
                "condense_s": _format_mean(_numeric_values(payloads, "condensation_time")),
                "train_s": _format_mean(_numeric_values(payloads, "training_time")),
                "infer_s": _format_mean(_numeric_values(payloads, "inference_time")),
                "peak_cpu_ram_gb": "" if first.get("peak_cpu_ram") in (None, "") else f"{float(first['peak_cpu_ram']) / 1e9:.6f}",
                "peak_gpu_ram_gb": "" if first.get("peak_gpu_ram") in (None, "") else f"{float(first['peak_gpu_ram']) / 1e9:.6f}",
                "status": first.get("status", "completed"),
                "source_log": ";".join(str(path.as_posix()) for path in paths),
            }
        )
    return sorted(rows, key=lambda row: (row["dataset"], row["method"], row["budget_mode"], row["ratio"], str(row["requested_target_budget"]), row["baseline_match_mode"]))


def build_ratio_budget_summary_rows(log_dirs: list[str | Path]) -> list[dict]:
    rows = []
    seen = set()
    for log_dir in log_dirs:
        for row in build_ratio_main_rows_from_logs(log_dir, recursive=True):
            key = (
                row["dataset"],
                row["method"],
                row["budget_mode"],
                row["ratio"],
                row["requested_target_budget"],
                row["effective_target_prototypes"],
                row["source_log"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({field: row.get(field, "") for field in RATIO_BUDGET_SUMMARY_FIELDS})
    return rows


def build_medium_main_rows_from_logs(log_dir: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(log_dir).glob("*.json")):
        if "_ks" in path.stem:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "generated_at": payload.get("generated_at", ""),
                "git_commit": payload.get("git_commit", ""),
                "config_hash": payload.get("config_hash", ""),
                "run_id": payload.get("run_id", ""),
                "dataset": payload.get("dataset", ""),
                "method": payload.get("method", payload.get("baseline", "Shadow-HGC-R-1")),
                "M_tau_requested": str(payload.get("requested_M_tau", payload.get("M_tau", ""))),
                "M_tau_effective": str(payload.get("effective_M_tau", "")),
                "feature_dim": str(payload.get("feature_dim", "")),
                "projection_type": payload.get("projection_type", ""),
                "loss_type": payload.get("loss_type", payload.get("ablation", {}).get("loss_type", "")),
                "model": payload.get("model", ""),
                "shadow_mode": payload.get("ablation", {}).get("shadow_mode", ""),
                "self_only": payload.get("ablation", {}).get("self_only", ""),
                "accuracy": "" if payload.get("accuracy") in (None, "") else f"{float(payload['accuracy']):.6f}",
                "macro_f1": "" if payload.get("macro_f1") in (None, "") else f"{float(payload['macro_f1']):.6f}",
                "num_predicted_classes": payload.get("num_predicted_classes", ""),
                "condensation_time": "" if payload.get("condensation_time") in (None, "") else f"{float(payload['condensation_time']):.6f}",
                "training_time": "" if payload.get("training_time") in (None, "") else f"{float(payload['training_time']):.6f}",
                "inference_time": "" if payload.get("inference_time") in (None, "") else f"{float(payload['inference_time']):.6f}",
                "status": payload.get("status", "completed"),
            }
        )
    return rows


def build_medium_ablation_rows_from_logs(log_dir: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(log_dir).glob("*_ks*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        setting = f"k_s={payload.get('k_s', path.stem.rsplit('_ks', 1)[-1])}"
        rows.append(
            {
                "dataset": payload.get("dataset", ""),
                "ablation": "target_target_skeleton",
                "setting": setting,
                "seed": payload.get("seed", ""),
                "accuracy": "" if payload.get("accuracy") in (None, "") else f"{float(payload['accuracy']):.6f}",
                "macro_f1": "" if payload.get("macro_f1") in (None, "") else f"{float(payload['macro_f1']):.6f}",
                "skeleton_coverage_mean": _mean_diag(payload, "SkeletonMassCoverage"),
                "residual_energy_mean": _mean_diag(payload, "ResidualEnergy"),
                "shadow_recon_err_mean": _mean_diag(payload, "ShadowReconErr"),
                "num_predicted_classes": payload.get("num_predicted_classes", ""),
                "prototype_train_acc": "" if payload.get("prototype_train_acc") in (None, "") else f"{float(payload['prototype_train_acc']):.6f}",
                "status": payload.get("status", "completed"),
            }
        )
    return rows


def write_rows_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = []
    for field in SMALL_MAIN_FIELDS + MEDIUM_MAIN_FIELDS + MEDIUM_ABLATION_FIELDS + RATIO_MAIN_FIELDS + RATIO_BUDGET_SUMMARY_FIELDS:
        if field not in preferred:
            preferred.append(field)
    fieldnames = [field for field in preferred if field in rows[0]]
    fieldnames += [field for field in rows[0] if field not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
