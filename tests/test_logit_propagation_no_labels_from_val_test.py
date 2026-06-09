from __future__ import annotations

import torch

from shadow_hgc.features.logit_propagation import propagate_logits


def test_logit_propagation_does_not_require_or_use_labels():
    result = propagate_logits(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        num_nodes=2,
        target_rows=torch.tensor([0, 1]),
        steps=1,
        lam=0.5,
        input_mode="probabilities",
    )

    assert result.diagnostics["uses_labels"] is False
    assert result.diagnostics["uses_validation_labels"] is False
    assert result.diagnostics["uses_test_labels"] is False
