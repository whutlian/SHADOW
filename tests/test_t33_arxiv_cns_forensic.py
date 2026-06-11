from __future__ import annotations

import torch

from shadow_hgc.sft.arxiv_cns_forensic_v4 import (
    arxiv_teacher_gate_reason,
    checksum_tensor,
    edge_direction_checksums,
    reject_historical_lad_logits,
)


def test_t33_arxiv_edge_direction_checksums_are_distinct() -> None:
    edge = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    checksums = edge_direction_checksums(edge)
    assert checksums["edge_checksum_cite_ref"] != checksums["edge_checksum_cited_by"]
    assert checksums["edge_checksum_undirected_sym"] not in {
        checksums["edge_checksum_cite_ref"],
        checksums["edge_checksum_cited_by"],
    }


def test_t33_arxiv_teacher_gate_reasons_are_explicit() -> None:
    assert arxiv_teacher_gate_reason(base_predictor="raw_x_mlp", cns_accuracy=0.63) == "cns_pipeline_mismatch_or_weak_base"
    assert arxiv_teacher_gate_reason(base_predictor="raw_x_mlp", cns_accuracy=0.705) == "teacher_gate_not_passed"
    assert arxiv_teacher_gate_reason(base_predictor="sagn_lite_v5", cns_accuracy=0.72) == ""


def test_t33_historical_lad_logits_rejected_for_main_rows() -> None:
    assert reject_historical_lad_logits("experiments/logits/t20_lad/raw_x_mlp_logits.pt") == "historical_lad_logits_not_allowed"
    assert reject_historical_lad_logits("experiments/logits/t33_arxiv/raw_x_mlp_logits.pt") == ""
    assert isinstance(checksum_tensor(torch.arange(4)), str)
