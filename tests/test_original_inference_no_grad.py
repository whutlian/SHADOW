import torch

from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.pipeline import core


def test_original_graph_inference_runs_without_grad(monkeypatch, tmp_path):
    graph = build_toy_graph(seed=0)
    grad_flags = []
    original_forward = core.WeightedRelationLinearConv.forward

    def recording_forward(self, *args, **kwargs):
        if kwargs.get("edge_chunk_size") is not None:
            grad_flags.append(torch.is_grad_enabled())
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(core.WeightedRelationLinearConv, "forward", recording_forward)

    core.run_shadow_hgc_experiment(
        graph,
        output_path=tmp_path / "summary.json",
        seed=0,
        epochs=1,
        M_tau=4,
        M_r=3,
        k_s=2,
        feature_dim=4,
        model_type="relation_linear",
        inference_edge_chunk_size=2,
    )

    assert grad_flags
    assert grad_flags[-1] is False
