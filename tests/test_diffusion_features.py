import torch

from shadow_hgc.features.diffusion import diffusion_target_features


def test_diffusion_feature_dimensions_are_correct_and_deterministic():
    x = torch.eye(3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    first = diffusion_target_features(
        x,
        edge_index=edge_index,
        num_nodes=3,
        steps=(1, 2),
        include_highpass=True,
    )
    second = diffusion_target_features(
        x,
        edge_index=edge_index,
        num_nodes=3,
        steps=(1, 2),
        include_highpass=True,
    )

    assert first.features.shape == (3, 9)
    assert first.block_names == ["X1", "X2", "Xhp"]
    assert torch.allclose(first.features, second.features)
