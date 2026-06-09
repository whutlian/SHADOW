from __future__ import annotations

from typing import Any


FORBIDDEN_T21_FLAGS = [
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_bounded_edges",
]


def validate_no_forbidden_t21_flags(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = [flag for flag in FORBIDDEN_T21_FLAGS if bool(row.get(flag, False))]
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def validate_products_full_execution_row(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("dataset", "")).lower() != "ogbn-products":
        return validate_no_forbidden_t21_flags(row)
    if str(row.get("status", "")) == "promoted" and bool(row.get("uses_bounded_edges", False)):
        raise ValueError("ogbn-products promoted full execution cannot use bounded edges")
    result = validate_no_forbidden_t21_flags(row)
    if str(row.get("status", "")) == "promoted" and not result["valid"]:
        raise ValueError(f"ogbn-products promoted full execution has forbidden flags: {result['forbidden_flags']}")
    return result
