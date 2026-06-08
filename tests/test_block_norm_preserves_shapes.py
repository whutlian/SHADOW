from __future__ import annotations

import torch

from shadow_hgc.features.block_norm import FeatureBlock, fit_transform_feature_blocks


def test_block_norm_preserves_block_shapes_and_names():
    x0 = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    degree = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]])
    blocks = [
        FeatureBlock("base:X0", x0, 4, "paper", "base"),
        FeatureBlock("degree", degree, 2, "paper", "degree"),
    ]

    transformed, stats = fit_transform_feature_blocks(
        blocks,
        fit_indices=torch.tensor([0, 1, 2]),
        mode="standardize",
    )

    assert list(transformed) == ["base:X0", "degree"]
    assert transformed["base:X0"].shape == x0.shape
    assert transformed["degree"].shape == degree.shape
    assert stats["base:X0"].name == "base:X0"
    assert stats["degree"].norm_p95 >= stats["degree"].norm_median
