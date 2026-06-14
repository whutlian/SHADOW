from __future__ import annotations

from pathlib import Path


def test_t39_runner_is_real_e2e_not_reference_fold() -> None:
    source = Path("scripts/run_t39_unified_e2e_stage.py").read_text(encoding="utf-8")

    assert "load_reference_index" not in source
    assert "current_sota_ratio_curve_summary" not in source
    assert "train_lazy_sft_from_memmap" in source
    assert "select_unified_prefixes_from_memmap" in source


def test_t39_runner_outputs_required_user_fields() -> None:
    source = Path("scripts/run_t39_unified_e2e_stage.py").read_text(encoding="utf-8")

    for field in ["budget_phase", "teacher_cache_k", "student_capacity", "post_cache_time_sec", "shared_cache_time_sec", "storage"]:
        assert field in source
