import torch

from shadow_hgc.data.ogb import build_homogeneous_special_case, trusted_ogb_torch_load


def test_homogeneous_special_case_builds_forward_and_reverse_relations():
    graph = build_homogeneous_special_case(
        dataset_name="toy-homo",
        target_type="paper",
        x=torch.eye(3),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        labels=torch.tensor([0, 1, 0]),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]),
        test_idx=torch.tensor([2]),
        forward_name="cite_ref",
        reverse_name="cited_by",
    )

    assert graph.target_type == "paper"
    assert len(graph.relations) == 2
    assert graph.relations[0].source_type == "paper"
    assert graph.relations[0].destination_type == "paper"
    assert torch.equal(graph.edge_index[graph.relations[1]], torch.tensor([[1, 2], [0, 1]]))


def test_trusted_ogb_torch_load_defaults_weights_only_false(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return "loaded"

    monkeypatch.setattr(torch, "load", fake_load)

    with trusted_ogb_torch_load():
        assert torch.load("processed.pt") == "loaded"

    assert calls == [{"weights_only": False}]
    assert torch.load is fake_load
