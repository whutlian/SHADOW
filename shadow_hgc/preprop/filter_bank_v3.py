from __future__ import annotations

from typing import Sequence

from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank
from shadow_hgc.preprop.manifest import PrepropManifest


T23_ARXIV_FILTER_BANK_V3_BLOCKS: tuple[str, ...] = (
    "X0",
    "X1_cite_ref",
    "X1_cited_by",
    "X2_cite_ref",
    "X2_cited_by",
    "X3_mix",
    "X4_mix",
    "Xres1_cite_ref",
    "Xres1_cited_by",
    "Xres2_mix",
    "Xres3_mix",
    "Y0_train_masked",
    "Y1_cite_ref",
    "Y1_cited_by",
    "Y2_cite_ref",
    "Y2_cited_by",
    "Y3_mix",
    "Y4_mix",
    "Yres1_mix",
    "structure",
)


def compute_t23_filter_bank_v3(
    *,
    dataset_name: str,
    graph_spec,
    feature_provider,
    target_node_ids,
    train_target_ids,
    labels,
    out_dir,
    blocks: Sequence[str] | None = None,
    feature_dim: int = 128,
    dtype: str = "float16",
    edge_chunk_size: int = 1_000_000,
    dst_chunk_size: int = 200_000,
) -> PrepropManifest:
    """T23 opt-in wrapper over the chunked destination-row filter bank.

    The underlying implementation writes fp16/fp32 memmaps and never builds an
    E-by-d edge feature tensor. This wrapper exists to make T23's block set and
    naming explicit without changing the default R-1 pipeline.
    """

    selected = tuple(blocks or T23_ARXIV_FILTER_BANK_V3_BLOCKS)
    return compute_preprop_filter_bank(
        dataset_name=dataset_name,
        graph_spec=graph_spec,
        feature_provider=feature_provider,
        target_node_ids=target_node_ids,
        train_target_ids=train_target_ids,
        labels=labels,
        out_dir=out_dir,
        blocks=selected,
        feature_dim=int(feature_dim),
        dtype=dtype,
        edge_chunk_size=int(edge_chunk_size),
        dst_chunk_size=int(dst_chunk_size),
        normalization="destination_row",
        fit_stats_on="train_target_rows",
    )
