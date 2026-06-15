from __future__ import annotations

from pathlib import Path

from scripts.run_t40_unified_auto_v2_stage import _candidate_schedule, _schedule_fields


def test_t40_runner_exposes_stage_outputs_and_fixed_method_id() -> None:
    source = Path("scripts/run_t40_unified_auto_v2_stage.py").read_text(encoding="utf-8")

    assert "shadow_stt_unified_auto_v2" in source
    assert "Shadow-HGC-STT-U" in source
    assert "t40_unified_auto_v2_main_curve_seed42.csv" in source
    assert "t40_unified_auto_v2_candidate_grid_seed42.csv" in source
    assert "t40_unified_auto_v2_gap_vs_reference_seed42.csv" in source
    assert "t40_unified_auto_v2_stage_summary.md" in source


def test_t40_runner_trains_candidates_instead_of_folding_reference_rows() -> None:
    source = Path("scripts/run_t40_unified_auto_v2_stage.py").read_text(encoding="utf-8")

    assert "train_lazy_sft_from_memmap" in source
    assert "select_best_candidate" in source
    assert "current_sota_ratio_curve_summary" in source
    assert "reference_accuracy" in source
    assert "method=\"products_uca_hybrid_mixup\"" not in source
    assert "method=\"reddit_ttcpp_gamlp_table_student\"" not in source


def test_t40_runner_has_required_output_fields() -> None:
    source = Path("scripts/run_t40_unified_auto_v2_stage.py").read_text(encoding="utf-8")

    for field in [
        "budget_phase",
        "class_capacity_b",
        "teacher_cache_k",
        "domain_coverage_gap",
        "policy_selection_score",
        "selected_policy",
        "student_capacity",
        "shared_cache_time_sec",
        "post_cache_time_sec",
        "storage",
    ]:
        assert field in source


def test_t40_schedule_fields_leave_runtime_safety_flags_to_runner() -> None:
    schedule = _candidate_schedule(
        "ogbn-papers100M",
        budget=111_060,
        teacher_valid_acc=0.63,
        domain_gap=0.2,
        args=type("Args", (), {"dense_cache_budget_mb": 256})(),
        policy="auto_base",
    )

    fields = _schedule_fields(schedule)

    assert "uses_dense_all_node_teacher_cache" not in fields
