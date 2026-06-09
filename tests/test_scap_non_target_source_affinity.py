from __future__ import annotations

import torch

from shadow_hgc.features.scap import non_target_source_scap, source_class_affinity


def test_scap_non_target_source_affinity_uses_active_sources_only():
    source_to_target = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]], dtype=torch.long)
    source_to_eval_target = torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long)
    labels = torch.tensor([0, 1, -1, -1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False])

    aff = source_class_affinity(
        source_to_target_edges=source_to_target,
        labels=labels,
        train_mask=train_mask,
        num_source_nodes=4,
        num_classes=2,
        active_source_nodes=torch.tensor([0, 1, 2]),
    )
    block, diagnostics = non_target_source_scap(
        edge_index_source_to_target=source_to_eval_target,
        source_affinity=aff,
        num_target_nodes=4,
        target_rows=torch.tensor([2, 3]),
    )

    assert block.shape == (2, 2)
    assert torch.allclose(block[0], torch.tensor([0.25, 0.75]), atol=1e-6)
    assert torch.allclose(block[1], torch.zeros(2), atol=1e-6)
    assert diagnostics["active_source_count"] == 3
    assert diagnostics["source_fallback_rate"] > 0.0
