from __future__ import annotations

from typing import Any


def sft_recovery_gap_row(
    *,
    dataset: str,
    ratio: float,
    method: str,
    fullgraph_accuracy: float,
    identity_accuracy: float,
    oracle_accuracy: float,
    shadow_accuracy: float,
    macro_f1: float = 0.0,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "ratio": float(ratio),
        "method": method,
        "status": status,
        "fullgraph_accuracy": float(fullgraph_accuracy),
        "identity_accuracy": float(identity_accuracy),
        "prototype_oracle_accuracy": float(oracle_accuracy),
        "shadow_accuracy": float(shadow_accuracy),
        "macro_f1": float(macro_f1),
        "full_to_identity_gap": float(fullgraph_accuracy) - float(identity_accuracy),
        "identity_to_oracle_gap": float(identity_accuracy) - float(oracle_accuracy),
        "oracle_to_shadow_gap": float(oracle_accuracy) - float(shadow_accuracy),
        "full_to_shadow_gap": float(fullgraph_accuracy) - float(shadow_accuracy),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
    }


def select_t23_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("accuracy", row.get("shadow_accuracy", "")) not in {"", None}]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row.get("accuracy", row.get("shadow_accuracy", 0.0)) or 0.0))
