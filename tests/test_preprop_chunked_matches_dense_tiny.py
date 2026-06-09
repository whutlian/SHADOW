import torch

from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.preprop.spmm_chunked import chunked_destination_spmm


def test_chunked_preprop_matches_dense_destination_normalized_tiny_graph():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [0, 0, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    source = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
        ]
    )
    alpha = destination_row_normalize(edge_index, num_dst_nodes=3).to(torch.float32)
    dense = torch.zeros(3, 2)
    dense.index_add_(0, edge_index[1], source[edge_index[0]] * alpha.unsqueeze(1))

    result = chunked_destination_spmm(
        edge_index=edge_index,
        source_features=source,
        num_dst_nodes=3,
        dst_rows=torch.tensor([0, 2]),
        edge_chunk_size=2,
    )

    torch.testing.assert_close(result.block, dense[[0, 2]])
    assert result.diagnostics["normalization"] == "destination_row"
