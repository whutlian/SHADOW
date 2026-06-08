from __future__ import annotations

import torch

from shadow_hgc.features.block_norm import FeatureBlock, fit_transform_feature_blocks


def test_block_norm_standardize_l2_no_nan_for_constant_blocks():
    blocks = [
        FeatureBlock(
            name="base:X0",
            tensor_or_provider=torch.ones(5, 3),
            dim=3,
            node_type="paper",
            role="base",
        ),
        FeatureBlock(
            name="diffusion:X1",
            tensor_or_provider=torch.zeros(5, 2),
            dim=2,
            node_type="paper",
            role="diffusion",
        ),
    ]

    transformed, stats = fit_transform_feature_blocks(
        blocks,
        fit_indices=torch.tensor([0, 2, 4]),
        mode="standardize_l2",
    )

    for tensor in transformed.values():
        assert torch.isfinite(tensor).all()
    assert torch.isfinite(stats["base:X0"].mean).all()
    assert torch.isfinite(stats["diffusion:X1"].std).all()
