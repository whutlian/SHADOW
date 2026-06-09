import torch

from shadow_hgc.logits.path_correct import PathStep, apply_path_logit_correct


def test_t1_path_logit_correct_streams_path_without_dense_path_edges():
    base = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target_to_source = PathStep(edge_index=torch.tensor([[0, 1], [0, 0]]), num_src=2, num_dst=1, name="movie_to_actor")
    source_to_target = PathStep(edge_index=torch.tensor([[0, 0], [0, 1]]), num_src=1, num_dst=2, name="actor_to_movie")

    result = apply_path_logit_correct(base_logits=base, steps=[target_to_source, source_to_target], alpha=1.0, space="prob")

    probs = torch.softmax(result.logits, dim=1)
    assert torch.allclose(probs[0], probs[1], atol=1e-6)
    assert result.diagnostics["uses_dense_path_adjacency"] is False
    assert result.diagnostics["exposes_metapath_edge_type"] is False
