from __future__ import annotations

from pathlib import Path

from scripts.run_t41_domain_transport_finalization import _candidate_schedule, _schedule_fields
from shadow_hgc.sft.t41_contract import FIXED_CANDIDATE_POLICIES, T41_MAIN_FIELDS


def test_t41_runner_exposes_stage_outputs_and_fixed_method_id() -> None:
    source = Path("scripts/run_t41_domain_transport_finalization.py").read_text(encoding="utf-8")

    assert "shadow_stt_unified_auto_v3" in source
    assert "Shadow-HGC-STT-U" in source
    assert "domain_transport" in source
    assert "t41_domain_transport_main_curve_seed42.csv" in source
    assert "t41_domain_transport_candidate_grid_seed42.csv" in source
    assert "t41_domain_transport_gap_vs_reference_seed42.csv" in source
    assert "t41_domain_transport_finalization_notes.md" in source


def test_t41_runner_uses_v3_selection_and_domain_transport_hook() -> None:
    source = Path("scripts/run_t41_domain_transport_finalization.py").read_text(encoding="utf-8")

    assert "apply_domain_transport_to_selection" in source
    assert "select_best_candidate" in source
    assert "score_v3" in source
    assert "domain_transport_gain" in source
    assert "method=\"products_uca_hybrid_mixup\"" not in source
    assert "if dataset == \"ogbn-products\"" not in source


def test_t41_runner_has_required_new_output_fields() -> None:
    source = Path("scripts/run_t41_domain_transport_finalization.py").read_text(encoding="utf-8")

    for field in [
        "budget_phase",
        "teacher_cache_k",
        "student_capacity",
        "shared_cache_time_sec",
        "post_cache_time_sec",
        "storage",
        "storage_bytes",
        "micro_f1",
        "domain_transport_active",
        "domain_transport_strength",
        "domain_transport_rows",
        "domain_row_frac",
        "domain_gap_before",
        "domain_gap_after",
        "domain_transport_gain",
        "domain_overfit_proxy",
        "row_type_counts",
    ]:
        assert field in T41_MAIN_FIELDS


def test_t41_candidate_schedule_accepts_domain_transport_policy() -> None:
    assert "domain_transport" in FIXED_CANDIDATE_POLICIES

    schedule = _candidate_schedule(
        "ogbn-products",
        budget=12_245,
        teacher_valid_acc=0.9,
        domain_gap=0.24,
        args=type("Args", (), {"dense_cache_budget_mb": 256})(),
        policy="domain_transport",
    )
    fields = _schedule_fields(schedule)

    assert schedule.policy_name == "domain_transport"
    assert fields["domain_transport_active"] is True
    assert "uses_dense_all_node_teacher_cache" not in fields


def test_t41_runner_no_cache_rebuild_flag_is_positive_guard() -> None:
    source = Path("scripts/run_t41_domain_transport_finalization.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--no-cache-rebuild", dest="no_cache_rebuild", action="store_true", default=True)' in source
    assert 'parser.add_argument("--allow-cache-rebuild", dest="no_cache_rebuild", action="store_false")' in source
    assert 'parser.add_argument("--no-cache-rebuild", action=argparse.BooleanOptionalAction' not in source
