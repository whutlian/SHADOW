from __future__ import annotations

from shadow_hgc.ultra.papers100m_contract import audit_cache_reuse, make_t35_row


def test_t35_reuse_audit_fails_when_ratio_rows_are_missing():
    audit = audit_cache_reuse([])

    assert audit["valid"] is False
    assert "ratio_rows_missing" in audit["failure_reasons"]


def test_t35_reuse_audit_blocks_mismatched_cache_ids_across_ratios():
    row_a = make_t35_row(requested_full_node_ratio=1e-5, condensed_nodes=1111, edge_slice_cache_id="edge-a", sft_cache_id="sft-a", teacher_cache_id="teacher-a", selection_bank_id="bank-a")
    row_b = make_t35_row(requested_full_node_ratio=1e-4, condensed_nodes=11106, edge_slice_cache_id="edge-a", sft_cache_id="sft-b", teacher_cache_id="teacher-a", selection_bank_id="bank-a")

    audit = audit_cache_reuse([row_a, row_b])

    assert audit["valid"] is False
    assert "sft_cache_id_mismatch" in audit["failure_reasons"]


def test_t35_reuse_audit_passes_identical_cache_ids_and_zero_incremental_scans():
    row_a = make_t35_row(requested_full_node_ratio=1e-5, condensed_nodes=1111, edge_slice_cache_id="edge-a", sft_cache_id="sft-a", teacher_cache_id="teacher-a", selection_bank_id="bank-a")
    row_b = make_t35_row(requested_full_node_ratio=1e-4, condensed_nodes=11106, edge_slice_cache_id="edge-a", sft_cache_id="sft-a", teacher_cache_id="teacher-a", selection_bank_id="bank-a")

    audit = audit_cache_reuse([row_a, row_b])

    assert audit["valid"] is True
    assert audit["failure_reasons"] == []
