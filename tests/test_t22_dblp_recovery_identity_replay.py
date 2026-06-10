from __future__ import annotations

import torch

from shadow_hgc.recovery.t22_sft_recovery import identity_replay_gap


def test_dblp_identity_replay_matches_fullgraph_predictions_in_tiny_mock():
    logits = torch.tensor([[3.0, 0.1], [0.2, 2.5], [1.1, 0.7]])
    replay_logits = logits.clone()
    labels = torch.tensor([0, 1, 0])
    gap = identity_replay_gap(full_logits=logits, replay_logits=replay_logits, labels=labels, rows=torch.arange(3))

    assert gap["accuracy_gap"] == 0.0
    assert gap["prediction_mismatch_rate"] == 0.0
