from __future__ import annotations

from typing import Any, Mapping

import torch

from shadow_hgc.preprop.label_reuse_v2 import compute_label_reuse_v2_blocks


def compute_label_reuse_v3_blocks(
    *,
    relation_blocks: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    train_target_ids: torch.Tensor,
    num_target_nodes: int,
    num_classes: int,
    edge_chunk_size: int = 1_000_000,
    prior_centering: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    blocks, diagnostics = compute_label_reuse_v2_blocks(
        relation_blocks=relation_blocks,
        labels=labels,
        train_target_ids=train_target_ids,
        num_target_nodes=int(num_target_nodes),
        num_classes=int(num_classes),
        edge_chunk_size=int(edge_chunk_size),
        prior_centering=bool(prior_centering),
    )
    if "Yres1_mix" in blocks and "Yres1" not in blocks:
        blocks["Yres1"] = blocks["Yres1_mix"]
    out_diag = {
        **diagnostics,
        "label_reuse_version": "v3",
        "uses_valid_labels": False,
        "uses_test_labels": False,
        "has_prior_centered_blocks": any(name.endswith("_prior_centered") for name in blocks),
    }
    return blocks, out_diag
