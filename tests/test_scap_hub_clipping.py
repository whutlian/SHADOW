from __future__ import annotations

import torch

from shadow_hgc.features.scap import apply_hub_clipping


def test_scap_hub_clipping_limits_source_degree_and_logs_fraction():
    edges = torch.tensor([[0, 0, 0, 1], [1, 2, 3, 3]], dtype=torch.long)

    clipped, meta = apply_hub_clipping(edges, hub_cap=2, policy="clip")

    assert clipped.shape[1] == 3
    assert int((clipped[0] == 0).sum().item()) == 2
    assert meta["num_hub_clipped_sources"] == 1
    assert meta["max_source_degree_before_clip"] == 3
    assert meta["max_source_degree_after_clip"] == 2
    assert meta["fraction_edges_clipped"] == 0.25
