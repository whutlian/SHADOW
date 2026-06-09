from __future__ import annotations

from typing import Any


FORBIDDEN_T2_PROMOTION_FLAGS = [
    "uses_logits_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_bounded_edges",
]


def validate_t2_promotion_row(row: dict[str, Any]) -> dict[str, Any]:
    invalid = []
    if row.get("status") == "promoted":
        for flag in FORBIDDEN_T2_PROMOTION_FLAGS:
            if bool(row.get(flag, False)):
                invalid.append(flag)
    return {"valid": len(invalid) == 0, "invalid_reasons": invalid}


def select_t2_safe_blocks(
    *,
    dataset: str,
    baseline: dict[str, float],
    candidates: list[dict[str, Any]],
    epsilon_acc: float = 1e-6,
    epsilon_f1: float = 1e-6,
) -> list[dict[str, Any]]:
    current_valid_acc = float(baseline.get("accuracy", 0.0))
    current_valid_f1 = float(baseline.get("macro_f1", 0.0))
    rows = []
    for candidate in candidates:
        valid_acc = float(candidate.get("valid_acc", 0.0))
        valid_f1 = float(candidate.get("valid_macro_f1", candidate.get("macro_f1", 0.0)))
        improves = valid_acc >= current_valid_acc + float(epsilon_acc) or valid_f1 >= current_valid_f1 + float(epsilon_f1)
        decision = "kept" if improves else "dropped"
        if improves:
            current_valid_acc = max(current_valid_acc, valid_acc)
            current_valid_f1 = max(current_valid_f1, valid_f1)
        rows.append(
            {
                "dataset": dataset,
                "block_group": candidate["block_group"],
                "branch_valid_acc": valid_acc,
                "branch_valid_macro_f1": valid_f1,
                "branch_test_acc_debug": float(candidate.get("test_acc", 0.0)),
                "gate_value": float(candidate.get("gate_value", 0.0)),
                "kept_or_dropped": decision,
                "drop_reason": "" if decision == "kept" else "dropped_by_validation",
                "block_dim": int(candidate.get("block_dim", 0)),
                "cache_bytes": int(candidate.get("cache_bytes", 0)),
                "uses_logits_as_input": False,
                "uses_dense_p2": False,
                "uses_e_by_d_materialization": False,
                "uses_bounded_edges": False,
            }
        )
    return rows
