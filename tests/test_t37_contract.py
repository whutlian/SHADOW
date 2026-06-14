from __future__ import annotations

from shadow_hgc.ultra.papers100m_t37_contract import (
    T37_DISCO_PARITY_RATIOS,
    attach_t37_reference_metrics,
    make_t37_row,
    summarize_t37_stage,
    validate_t37_row,
)


def test_t37_rejects_promoted_disco_rows_without_sgc_backend():
    row = make_t37_row(
        method="scr_class_random",
        backend="gamlp_table",
        comparison_type="disco_parity",
        requested_full_node_ratio=T37_DISCO_PARITY_RATIOS[0],
        promotion_status="promoted",
    )

    result = validate_t37_row(row)

    assert result["valid"] is False
    assert "disco_parity_backend_not_sgc" in result["forbidden_flags"]


def test_t37_rejects_promoted_forbidden_paths_and_label_leakage():
    row = make_t37_row(
        method="scr_full_stochastic_coverage",
        backend="sgc",
        comparison_type="disco_parity",
        requested_full_node_ratio=T37_DISCO_PARITY_RATIOS[0],
        promotion_status="promoted",
        uses_dense_all_node_teacher_cache=True,
        uses_valid_labels_as_input=True,
        incremental_edge_scans_after_cache_build=1,
    )

    result = validate_t37_row(row)

    assert result["valid"] is False
    assert "uses_dense_all_node_teacher_cache" in result["forbidden_flags"]
    assert "uses_valid_labels_as_input" in result["forbidden_flags"]
    assert "incremental_edge_scans_after_cache_build_nonzero" in result["forbidden_flags"]


def test_t37_reference_metrics_include_random_and_relative_error_reduction():
    row = make_t37_row(
        method="scr_class_random",
        backend="sgc",
        comparison_type="disco_parity",
        requested_full_node_ratio=0.0001,
        accuracy=0.50,
    )
    refs = {0.0001: {"disco_acc": 0.487, "random_onecache_acc": 0.499}}

    out = attach_t37_reference_metrics(row, refs)

    assert out["beats_disco"] is True
    assert out["beats_random_onecache"] is True
    assert abs(out["relative_error_reduction_vs_disco"] - ((0.50 - 0.487) / (1.0 - 0.487))) < 1e-12
    assert abs(out["relative_error_reduction_vs_random"] - ((0.50 - 0.499) / (1.0 - 0.499))) < 1e-12


def test_t37_stage_summary_counts_stop_conditions():
    rows = [
        make_t37_row(
            method="scr_class_random",
            backend="sgc",
            comparison_type="disco_parity",
            requested_full_node_ratio=ratio,
            accuracy=acc,
            disco_acc=disco,
            random_onecache_acc=random,
            beats_disco=acc >= disco,
            beats_random_onecache=acc >= random,
            promotion_status="promoted",
        )
        for ratio, acc, disco, random in [
            (0.00005, 0.484, 0.483, 0.448),
            (0.00010, 0.505, 0.487, 0.499),
            (0.00020, 0.537, 0.496, 0.531),
            (0.00050, 0.573, 0.509, 0.566),
        ]
    ]

    summary = summarize_t37_stage(disco_rows=rows, native_rows=[], multiseed_rows=[], bank_rows=[], teacher_rows=[])

    assert summary["disco_beats_disco_count"] == 4
    assert summary["disco_beats_random_count"] == 4
    assert summary["disco_success_gate"] is True
    assert summary["forbidden_guard_hits"] == 0
