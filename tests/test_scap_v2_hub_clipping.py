from __future__ import annotations

import torch

from shadow_hgc.features.scap_v2 import clip_source_hubs


def test_scap_v2_hub_clipping_logs_clipped_hubs():
    edge_index = torch.tensor([[0, 0, 0, 1], [0, 1, 2, 2]], dtype=torch.long)

    clipped, diagnostics = clip_source_hubs(edge_index, hub_cap=2)

    assert clipped.shape[1] == 3
    assert diagnostics["hub_cap"] == 2
    assert diagnostics["num_clipped_hubs"] == 1
    assert diagnostics["max_source_degree_after_clip"] == 2
