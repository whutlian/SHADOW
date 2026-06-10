from __future__ import annotations

from typing import Any


FORBIDDEN_T22_FLAGS = [
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def validate_t22_promoted_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = [flag for flag in FORBIDDEN_T22_FLAGS if _truthy(row.get(flag, False))]
    missing = [field for field in ["dataset", "accuracy", "macro_f1", "predicted_class_count"] if field not in row]
    if str(row.get("dataset", "")).startswith("ogbn") and not _truthy(row.get("uses_memmap", True)):
        forbidden.append("missing_memmap_for_ogb")
    if str(row.get("dataset", "")).startswith("ogbn") and not _truthy(row.get("full_edge_execution", True)):
        forbidden.append("not_full_edge_execution")
    return {"valid": not forbidden and not missing, "forbidden_flags": forbidden, "missing_fields": missing}
