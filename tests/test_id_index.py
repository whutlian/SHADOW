import torch

from shadow_hgc.data.id_index import IdIndex


def test_id_index_uses_dense_int32_when_within_budget():
    index = IdIndex.build(
        torch.tensor([2, 5, 9], dtype=torch.long),
        num_nodes=10,
        dense_map_budget_bytes=40,
    )

    assert index.mode == "dense_int32"
    assert index.storage_nbytes == 40
    assert torch.equal(
        index.lookup(torch.tensor([5, 0, 9, 2, 10, -1], dtype=torch.long)),
        torch.tensor([1, -1, 2, 0, -1, -1], dtype=torch.long),
    )


def test_id_index_uses_sorted_search_when_dense_map_exceeds_budget():
    index = IdIndex.build(
        torch.tensor([2, 5, 9], dtype=torch.long),
        num_nodes=1_000_000_000,
        dense_map_budget_bytes=16,
    )

    assert index.mode == "sorted_search"
    assert index.storage_nbytes < 1_000
    assert torch.equal(
        index.lookup(torch.tensor([9, 3, 2, 5], dtype=torch.long)),
        torch.tensor([2, -1, 0, 1], dtype=torch.long),
    )


def test_id_index_rejects_duplicate_index_ids():
    try:
        IdIndex.build(torch.tensor([1, 1], dtype=torch.long), num_nodes=4)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate ids should be rejected")
