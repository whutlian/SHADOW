from __future__ import annotations

from typing import Sequence

from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank
from shadow_hgc.preprop.manifest import PrepropManifest


T24_ARXIV_FILTER_BANK_V4_BLOCKS: tuple[str, ...] = (
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


def t24_arxiv_v4_blocks(
    *,
    include_x4_mix: bool = True,
    include_xres2: bool = True,
    include_xres3: bool = True,
    include_symnorm_ablation: bool = False,
    include_y0: bool = True,
    include_y4: bool = True,
    include_yres1: bool = True,
) -> tuple[str, ...]:
    blocks: list[str] = [
        "X0",
        "X1_cite_ref",
        "X1_cited_by",
        "X2_cite_ref",
        "X2_cited_by",
        "X3_mix",
    ]
    if include_x4_mix:
        blocks.append("X4_mix")
    blocks.extend(["Xres1_cite_ref", "Xres1_cited_by"])
    if include_xres2:
        blocks.append("Xres2_mix")
    if include_xres3:
        blocks.append("Xres3_mix")
    if include_symnorm_ablation:
        blocks.extend(["X1_sym", "X2_sym"])
    if include_y0:
        blocks.append("Y0_train_masked")
    blocks.extend(["Y1_cite_ref", "Y1_cited_by", "Y2_cite_ref", "Y2_cited_by", "Y3_mix"])
    if include_y4:
        blocks.append("Y4_mix")
    if include_yres1:
        blocks.append("Yres1_mix")
    blocks.append("structure")
    return tuple(blocks)


def compute_t24_filter_bank_v4(
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
    if int(feature_dim) > 128:
        raise ValueError("T24 arxiv filter-bank v4 requires block_dim <= 128")
    selected = tuple(blocks or T24_ARXIV_FILTER_BANK_V4_BLOCKS)
    unsupported = [name for name in selected if name in {"X1_sym", "X2_sym"}]
    if unsupported:
        raise ValueError("symmetric-normalization ablation names are CLI-visible but not promoted in the v4 row-normalized path")
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
