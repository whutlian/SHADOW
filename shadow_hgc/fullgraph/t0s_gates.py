from __future__ import annotations

from typing import Any


T0S_ACCURACY_GATES: dict[str, float] = {
    "acm": 0.930,
    "dblp": 0.910,
    "imdb": 0.600,
    "ogbn-arxiv": 0.700,
    "ogbn-products": 0.720,
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "passed"}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_t0s_row(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate T0-S accuracy and scalability gates for a result row."""
    dataset = str(row.get("dataset", "")).lower()
    gate_acc = T0S_ACCURACY_GATES.get(dataset)
    accuracy = _as_float(row.get("accuracy"))
    gate_acc_passed = bool(gate_acc is not None and accuracy is not None and accuracy >= gate_acc)

    blocked: list[str] = []
    for field in ["uses_diffusion", "uses_dense_p2", "uses_full_graph_backprop"]:
        if _as_bool(row.get(field)):
            blocked.append(field)
    if not _as_bool(row.get("train_label_only", True)):
        blocked.append("not_train_label_only")
    if _as_float(row.get("full_edge_scans"), 0.0) is None:
        blocked.append("missing_full_edge_scans")
    if _as_float(row.get("cache_bytes"), 0.0) is None:
        blocked.append("missing_cache_bytes")
    if _as_float(row.get("peak_cpu_ram_gb"), 0.0) is None:
        blocked.append("missing_peak_cpu_ram_gb")
    if _as_float(row.get("peak_gpu_ram_gb"), 0.0) is None:
        blocked.append("missing_peak_gpu_ram_gb")
    if _as_bool(row.get("cache_all_targets")):
        blocked.append("cache_all_targets_forbidden")
    if _as_bool(row.get("uses_dense_e_by_d")):
        blocked.append("uses_dense_e_by_d")

    out = dict(row)
    out["gate_acc"] = gate_acc
    out["gate_acc_passed"] = gate_acc_passed
    out["gate_scalability_passed"] = len(blocked) == 0
    out["blocked_reason"] = ";".join(blocked)
    out["gate_passed"] = gate_acc_passed and len(blocked) == 0
    return out


def required_scalability_fields() -> list[str]:
    return [
        "num_nodes",
        "num_edges",
        "num_target_rows",
        "num_train_target_rows",
        "num_active_sources",
        "num_classes",
        "feature_dim",
        "scap_topk",
        "full_edge_scans",
        "peak_cpu_ram_gb",
        "peak_gpu_ram_gb",
        "disk_cache_gb",
        "scap_cache_gb",
        "feature_demand_cache_gb",
        "wall_time_s",
        "edge_scan_throughput_edges_per_s",
        "cache_all_targets",
        "uses_dense_e_by_d",
    ]


def validate_scalability_resource_row(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for field in required_scalability_fields():
        if field not in row:
            reasons.append(f"missing_{field}")
    if _as_bool(row.get("cache_all_targets")):
        reasons.append("cache_all_targets_forbidden")
    if _as_bool(row.get("uses_dense_e_by_d")):
        reasons.append("uses_dense_e_by_d_forbidden")
    if _as_float(row.get("full_edge_scans"), 0.0) is not None and float(row.get("full_edge_scans", 0)) > 2:
        reasons.append("too_many_full_edge_scans")
    return {"valid": len(reasons) == 0, "reasons": reasons}
