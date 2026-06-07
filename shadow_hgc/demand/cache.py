from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass
class EdgeSliceCache:
    dst_train_pos: torch.Tensor
    src_train_pos: torch.Tensor
    alpha: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.alpha.numel())

    @property
    def nbytes(self) -> int:
        return int(
            self.dst_train_pos.numel() * self.dst_train_pos.element_size()
            + self.src_train_pos.numel() * self.src_train_pos.element_size()
            + self.alpha.numel() * self.alpha.element_size()
        )


@dataclass
class RelationCacheStats:
    full_edge_scans: int
    active_source_count: int
    edge_slice_cache_bytes: int


@dataclass
class RelationDemandCache:
    demand: torch.Tensor
    degree_weight_sum: torch.Tensor
    active_sources: torch.Tensor
    edge_slice: EdgeSliceCache | None
    stats: RelationCacheStats


def validate_train_target_only_cache(
    *,
    num_target_nodes: int,
    train_target_ids: torch.Tensor,
    cache_all_targets: bool,
    debug_allow_all_node_cache: bool = False,
) -> None:
    """Large mode may not silently allocate all-node relation demand."""

    if cache_all_targets and train_target_ids.numel() >= num_target_nodes and not debug_allow_all_node_cache:
        raise ValueError("large-mode all-node relation demand cache is forbidden outside debug mode")


def estimate_ultra_dry_run(
    *,
    num_train_targets: int,
    num_relations: int,
    feature_dim: int,
    active_source_count: int,
    train_train_edges: int,
    dtype_bytes: int = 4,
) -> dict[str, int | bool]:
    demand_cache_bytes = num_train_targets * num_relations * feature_dim * dtype_bytes
    edge_slice_cache_bytes = train_train_edges * (4 + 4 + 4)
    active_source_feature_bytes = active_source_count * feature_dim * dtype_bytes
    return {
        "demand_cache_bytes": int(demand_cache_bytes),
        "edge_slice_cache_bytes": int(edge_slice_cache_bytes),
        "active_source_feature_bytes": int(active_source_feature_bytes),
        "expected_full_edge_scans": int(2 * num_relations),
        "peak_ram_estimate_bytes": int(demand_cache_bytes + edge_slice_cache_bytes + active_source_feature_bytes),
        "disk_spill_estimate_bytes": int(edge_slice_cache_bytes),
        "disk_spill_used": bool(edge_slice_cache_bytes > 2**31),
    }


def _make_id_to_pos(ids: torch.Tensor, size: int) -> torch.Tensor:
    mapping = torch.full((size,), -1, dtype=torch.long)
    mapping[ids.to(torch.long)] = torch.arange(ids.numel(), dtype=torch.long)
    return mapping


def build_relation_demand_cache(
    *,
    edge_stream_factory: Callable[[], Iterable],
    train_target_ids: torch.Tensor,
    num_target_nodes: int,
    num_source_nodes: int,
    source_feature_getter: Callable[[torch.Tensor], torch.Tensor],
    feature_dim: int,
    source_is_target: bool,
    cache_edge_slice: bool = False,
    debug_allow_all_node_cache: bool = False,
) -> RelationDemandCache:
    """Two-pass train-target-only streaming demand cache."""

    validate_train_target_only_cache(
        num_target_nodes=num_target_nodes,
        train_target_ids=train_target_ids,
        cache_all_targets=train_target_ids.numel() >= num_target_nodes,
        debug_allow_all_node_cache=debug_allow_all_node_cache,
    )
    train_target_ids = train_target_ids.to(torch.long).cpu()
    dst_to_pos = _make_id_to_pos(train_target_ids, num_target_nodes)
    src_train_to_pos = _make_id_to_pos(train_target_ids, num_source_nodes) if source_is_target else None

    degree_weight_sum = torch.zeros(train_target_ids.numel(), dtype=torch.float32)
    active_mask = torch.zeros(num_source_nodes, dtype=torch.bool)
    full_edge_scans = 0

    for chunk in edge_stream_factory():
        dst_pos = dst_to_pos[chunk.dst.cpu()]
        mask = dst_pos >= 0
        if bool(mask.any()):
            degree_weight_sum.index_add_(0, dst_pos[mask], chunk.weight[mask].cpu())
            active_mask[chunk.src[mask].cpu()] = True
    full_edge_scans += 1

    demand = torch.zeros(train_target_ids.numel(), feature_dim, dtype=torch.float32)
    edge_dst: list[torch.Tensor] = []
    edge_src: list[torch.Tensor] = []
    edge_alpha: list[torch.Tensor] = []

    for chunk in edge_stream_factory():
        dst_pos = dst_to_pos[chunk.dst.cpu()]
        mask = dst_pos >= 0
        if not bool(mask.any()):
            continue
        selected_dst = dst_pos[mask]
        selected_src = chunk.src[mask].cpu()
        selected_weight = chunk.weight[mask].cpu()
        alpha = selected_weight / degree_weight_sum[selected_dst].clamp_min(1e-12)
        source_features = source_feature_getter(selected_src).to(torch.float32).cpu()
        demand.index_add_(0, selected_dst, source_features * alpha.unsqueeze(-1))

        if cache_edge_slice and source_is_target and src_train_to_pos is not None:
            selected_src_pos = src_train_to_pos[selected_src]
            slice_mask = selected_src_pos >= 0
            if bool(slice_mask.any()):
                edge_dst.append(selected_dst[slice_mask].to(torch.int32))
                edge_src.append(selected_src_pos[slice_mask].to(torch.int32))
                edge_alpha.append(alpha[slice_mask].to(torch.float32))
    full_edge_scans += 1

    edge_slice = None
    if cache_edge_slice and edge_alpha:
        edge_slice = EdgeSliceCache(
            dst_train_pos=torch.cat(edge_dst),
            src_train_pos=torch.cat(edge_src),
            alpha=torch.cat(edge_alpha),
        )
    active_sources = torch.nonzero(active_mask, as_tuple=False).flatten().to(torch.long)
    stats = RelationCacheStats(
        full_edge_scans=full_edge_scans,
        active_source_count=int(active_sources.numel()),
        edge_slice_cache_bytes=0 if edge_slice is None else edge_slice.nbytes,
    )
    return RelationDemandCache(
        demand=demand,
        degree_weight_sum=degree_weight_sum,
        active_sources=active_sources,
        edge_slice=edge_slice,
        stats=stats,
    )
