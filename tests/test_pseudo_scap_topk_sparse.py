import torch

from shadow_hgc.features.pseudo_scap import dense_to_topk_sparse


def test_pseudo_scap_topk_sparse_keeps_largest_classes():
    values = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1]])

    sparse = dense_to_topk_sparse(values, topk=2)

    assert sparse.indices.shape == (2, 2)
    assert sparse.values.shape == (2, 2)
    assert sparse.indices[0].tolist() == [1, 2]
    assert torch.allclose(sparse.values[0], torch.tensor([0.7, 0.2]))
