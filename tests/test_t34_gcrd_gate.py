from __future__ import annotations

from shadow_hgc.sft.gcrd_gate import compute_gcrd_gate_row


def test_t34_gcrd_relative_error_reduction_formula() -> None:
    row = compute_gcrd_gate_row(dataset="Reddit", ratio=0.001, ours_method="m", ours_acc=0.905, baseline_acc=0.90)
    assert abs(row["relative_error_reduction"] - 0.05) < 1e-12
    assert row["passes_5pct_error_reduction"] is True
    row2 = compute_gcrd_gate_row(dataset="ogbn-products", ratio=0.005, ours_method="m", ours_acc=0.81, baseline_acc=0.80)
    assert abs(row2["relative_error_reduction"] - 0.05) < 1e-12


def test_t34_gcrd_logs_impossible_absolute_gate_under_teacher_ceiling() -> None:
    row = compute_gcrd_gate_row(
        dataset="Reddit",
        ratio=0.005,
        ours_method="m",
        ours_acc=0.939,
        baseline_acc=0.91,
        teacher_ceiling_acc=0.94,
    )
    assert row["passes_absolute_5pp_if_applicable"] is False
    assert row["mathematically_impossible_under_current_teacher_ceiling"] is True
    assert row["teacher_ceiling_gap"] > 0.0


def test_t34_gcrd_todo_baseline_rows_do_not_fabricate() -> None:
    row = compute_gcrd_gate_row(dataset="ogbn-arxiv", ratio=0.005, ours_method="m", ours_acc=0.0, baseline_acc="TODO_EXACT_VALUE")
    assert row["baseline_acc"] == "TODO_EXACT_VALUE"
    assert row["passes_5pct_error_reduction"] is False
    assert row["notes"] == "manual_input_required"
