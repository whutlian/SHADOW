from __future__ import annotations

import torch

from shadow_hgc.recovery.sft_signature import SFTBlockSignature, build_recovery_signature


def test_sft_block_signature_keeps_selected_blocks_and_frozen_train_stats():
    full = SFTBlockSignature(
        dataset="dblp",
        selected_blocks=["self", "typed:writes"],
        block_stats={"self": {"source": "train_target_rows", "frozen": True, "fit_rows": [0, 1]}},
        logits=torch.zeros(3, 2),
        labels=torch.tensor([0, 1, 0]),
        num_classes=2,
    )
    recovery = build_recovery_signature(full, condensed_row_map=torch.tensor([0, 2], dtype=torch.long), recovery_kind="prototype_oracle")

    assert recovery.dataset == "dblp"
    assert recovery.selected_blocks == full.selected_blocks
    assert recovery.block_stats["self"]["source"] == "train_target_rows"
    assert recovery.block_stats["self"]["frozen"] is True
    assert recovery.recovery_kind == "prototype_oracle"
