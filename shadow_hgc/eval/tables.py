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


def _load_json_logs(log_dir: str | Path) -> list[dict]:
    path = Path(log_dir)
    return [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]


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
    for field in SMALL_MAIN_FIELDS + MEDIUM_MAIN_FIELDS + MEDIUM_ABLATION_FIELDS:
        if field not in preferred:
            preferred.append(field)
    fieldnames = [field for field in preferred if field in rows[0]]
    fieldnames += [field for field in rows[0] if field not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
