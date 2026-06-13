from __future__ import annotations

from typing import Any


def _is_todo(value: Any) -> bool:
    return str(value) == "TODO_EXACT_VALUE" or value in {"", None}


def compute_gcrd_gate_row(
    *,
    dataset: str,
    ratio: float,
    ours_method: str,
    ours_acc: float,
    baseline_acc: float | str,
    baseline_std: float | str = "",
    teacher_ceiling_acc: float | str = "",
    ratio_definition_match: bool = True,
) -> dict[str, Any]:
    if _is_todo(baseline_acc):
        return {
            "dataset": dataset,
            "ratio": float(ratio),
            "ours_method": ours_method,
            "ours_acc": float(ours_acc),
            "baseline_acc": "TODO_EXACT_VALUE",
            "gcrd_accuracy_std": baseline_std,
            "absolute_pp_gain": "",
            "relative_accuracy_gain": "",
            "relative_error_reduction": "",
            "passes_5pct_error_reduction": False,
            "passes_absolute_5pp_if_applicable": False,
            "teacher_ceiling_gap": "",
            "ratio_definition_match": bool(ratio_definition_match),
            "mathematically_impossible_under_current_teacher_ceiling": False,
            "notes": "manual_input_required",
        }
    base = float(baseline_acc)
    ours = float(ours_acc)
    abs_gain = ours - base
    rel_acc = abs_gain / base if base else 0.0
    rel_err = abs_gain / max(1e-12, 1.0 - base)
    impossible = False
    ceiling_gap: float | str = ""
    if not _is_todo(teacher_ceiling_acc):
        ceiling = float(teacher_ceiling_acc)
        ceiling_gap = max(0.0, ceiling - ours)
        impossible = (base + 0.05) > ceiling
    return {
        "dataset": dataset,
        "ratio": float(ratio),
        "ours_method": ours_method,
        "ours_acc": ours,
        "baseline_acc": base,
        "gcrd_accuracy_std": baseline_std,
        "absolute_pp_gain": abs_gain,
        "relative_accuracy_gain": rel_acc,
        "relative_error_reduction": rel_err,
        "passes_5pct_error_reduction": rel_err >= 0.05,
        "passes_absolute_5pp_if_applicable": abs_gain >= 0.05,
        "teacher_ceiling_gap": ceiling_gap,
        "ratio_definition_match": bool(ratio_definition_match),
        "mathematically_impossible_under_current_teacher_ceiling": impossible,
        "notes": "",
    }
