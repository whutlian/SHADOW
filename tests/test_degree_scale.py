import torch

from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.pipeline.core import prepare_model_features


def test_degree_scale_controls_target_model_input_degree_block():
    graph = build_toy_graph()

    _, phi_small, degree_small, _ = prepare_model_features(
        graph,
        feature_dim=4,
        seed=0,
        projection_type="random",
        degree_scale=0.1,
    )
    _, phi_large, degree_large, _ = prepare_model_features(
        graph,
        feature_dim=4,
        seed=0,
        projection_type="random",
        degree_scale=1.0,
    )

    assert torch.allclose(degree_small, degree_large)
    assert torch.allclose(phi_small["paper"][:, -degree_small.shape[1] :] * 10.0, phi_large["paper"][:, -degree_large.shape[1] :])


def test_raw_projection_type_preserves_raw_feature_dimension_after_standardization():
    graph = build_toy_graph()

    psi, phi, degree_features, _ = prepare_model_features(
        graph,
        feature_dim=16,
        seed=0,
        projection_type="raw",
        degree_scale=0.1,
    )

    assert psi["paper"].shape[1] == graph.node_features["paper"].shape[1]
    assert phi["paper"].shape[1] == graph.node_features["paper"].shape[1] + degree_features.shape[1]
