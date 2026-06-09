from __future__ import annotations

import torch

from shadow_hgc.recovery.sft_signature import SFTBlockSignature, identity_replay


def test_sft_condensation_identity_replay_preserves_logits_and_metrics():
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.5]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    rows = torch.tensor([0, 1, 2], dtype=torch.long)
    signature = SFTBlockSignature(
        dataset="tiny",
        selected_blocks=["self", "typed:cite"],
        block_stats={"self": {"source": "train_target_rows", "fit_rows": [0, 1]}},
        logits=logits,
        labels=labels,
        num_classes=2,
    )
    replay = identity_replay(signature, rows=rows)

    assert torch.equal(replay.logits, logits)
    assert replay.metrics["accuracy"] == 1.0
    assert replay.full_to_identity_gap == 0.0
    assert replay.uses_logits_as_input is False
