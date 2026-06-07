from shadow_hgc.baselines.full_graph_same_backbone import run_full_graph_same_backbone
from shadow_hgc.data.loaders import build_toy_graph


def test_full_graph_same_backbone_baseline_uses_original_graph(tmp_path):
    graph = build_toy_graph()

    summary = run_full_graph_same_backbone(
        graph,
        output_path=tmp_path / "full.json",
        seed=0,
        epochs=2,
        feature_dim=4,
        projection_type="random",
        degree_scale=0.1,
        model_type="relation_linear",
    )

    assert summary["baseline"] == "Full-WRL-GNN"
    assert summary["accuracy"] is not None
    assert summary["num_optimizer_steps"] == 2
    assert summary["directed_relations"]
