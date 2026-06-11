from __future__ import annotations

import torch

from shadow_hgc.sft.qoc_forensic import assignment_hash, assignment_overlap, build_qoc_forensic_rows


def test_t31_qoc_forensic_produces_identity_and_operator_rows() -> None:
    rows = build_qoc_forensic_rows(dataset="Reddit", seed=42, ratio=0.001, num_codewords=4, reference_acc=0.9)
    modes = {row["forensic_mode"] for row in rows}
    assert {"identity", "table_only", "pz_only", "pz_p2z"}.issubset(modes)
    assert all(row["promotion_status"] == "not_promoted" for row in rows)


def test_t31_qoc_assignment_hash_and_overlap_detect_distinct_assignments() -> None:
    a = torch.tensor([0, 0, 1, 1, 2, 2])
    b = torch.tensor([0, 1, 1, 2, 2, 0])
    assert assignment_hash(a) != assignment_hash(b)
    assert assignment_overlap(a, b) < 0.95


def test_t31_qoc_identity_failure_blocks_promotion() -> None:
    rows = build_qoc_forensic_rows(dataset="Reddit", seed=42, ratio=0.001, num_codewords=4, reference_acc=0.9, identity_acc=0.5)
    identity = [row for row in rows if row["forensic_mode"] == "identity"][0]
    assert identity["status"] == "completed_forensic"
    assert identity["failure_reason"] == "identity_transfer_below_reference"
