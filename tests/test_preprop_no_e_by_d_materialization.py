import torch

from shadow_hgc.preprop.spmm_chunked import chunked_destination_spmm


def test_chunked_preprop_reports_no_full_e_by_d_materialization():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2],
            [0, 0, 1, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    source = torch.randn(3, 4)
    result = chunked_destination_spmm(
        edge_index=edge_index,
        source_features=source,
        num_dst_nodes=3,
        dst_rows=torch.arange(3),
        edge_chunk_size=2,
    )

    assert result.diagnostics["uses_e_by_d_materialization"] is False
    assert result.diagnostics["materialized_full_e_by_d"] is False
    assert result.diagnostics["max_edge_chunk_size"] == 2
    assert result.block.shape == (3, 4)
