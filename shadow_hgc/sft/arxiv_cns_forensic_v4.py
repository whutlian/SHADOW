from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from shadow_hgc.sft.t32_arxiv_cns import transform_arxiv_edge_index


def checksum_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def checksum_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def edge_direction_checksums(edge_index: torch.Tensor) -> dict[str, str]:
    cite_ref = transform_arxiv_edge_index(edge_index, graph_direction="cite_ref", self_loop_mode="none")
    cited_by = transform_arxiv_edge_index(edge_index, graph_direction="cited_by", self_loop_mode="none")
    undirected = transform_arxiv_edge_index(edge_index, graph_direction="undirected_sym", self_loop_mode="none")
    return {
        "edge_checksum_cite_ref": checksum_tensor(cite_ref),
        "edge_checksum_cited_by": checksum_tensor(cited_by),
        "edge_checksum_undirected_sym": checksum_tensor(undirected),
    }


def reject_historical_lad_logits(path: str | Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "lad" in text or "historical" in text:
        return "historical_lad_logits_not_allowed"
    return ""


def arxiv_teacher_gate_reason(*, base_predictor: str, cns_accuracy: float | str) -> str:
    try:
        acc = float(cns_accuracy)
    except (TypeError, ValueError):
        return "missing_cns_accuracy"
    if str(base_predictor) == "raw_x_mlp":
        if acc < 0.700:
            return "cns_pipeline_mismatch_or_weak_base"
        if acc < 0.725:
            return "teacher_gate_not_passed"
        return ""
    if acc < 0.715:
        return "teacher_gate_not_passed"
    return ""


def arxiv_forensic_payload(
    *,
    num_nodes: int,
    num_edges: int,
    num_classes: int,
    train_count: int,
    valid_count: int,
    test_count: int,
    feature_shape: tuple[int, int],
    feature_checksum: str,
    label_checksum: str,
    train_mask_checksum: str,
    valid_mask_checksum: str,
    test_mask_checksum: str,
    edge_checksums: dict[str, str],
) -> dict[str, Any]:
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_classes": int(num_classes),
        "train_count": int(train_count),
        "valid_count": int(valid_count),
        "test_count": int(test_count),
        "feature_shape": list(feature_shape),
        "feature_checksum": feature_checksum,
        "label_checksum": label_checksum,
        "train_mask_checksum": train_mask_checksum,
        "valid_mask_checksum": valid_mask_checksum,
        "test_mask_checksum": test_mask_checksum,
        **edge_checksums,
    }
