from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import torch

from shadow_hgc.data.id_index import IdIndex, select_id_index_mode


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

    @property
    def dtype_summary(self) -> str:
        return f"{self.dst_train_pos.dtype},{self.src_train_pos.dtype},{self.alpha.dtype}".replace("torch.", "")


@dataclass
class RelationCacheStats:
    full_edge_scans: int
    active_source_count: int
    edge_slice_cache_bytes: int
    edge_slice_cache_edges: int = 0
    edge_slice_dtype: str = "none"
    cache_build_time: float = 0.0
    cache_aggregation_time: float = 0.0
    disk_spill_used: bool = False
    dst_id_index_mode: str = ""
    src_train_id_index_mode: str | None = None
    dst_id_index_bytes: int = 0
    src_train_id_index_bytes: int = 0


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

    if cache_all_targets and not debug_allow_all_node_cache:
        raise ValueError("large-mode all-node relation demand cache is forbidden outside debug mode")


def estimate_ultra_dry_run(
    *,
    num_train_targets: int,
    feature_dim: int,
    num_relations: int | None = None,
    active_source_count: int | None = None,
    train_train_edges: int | None = None,
    dtype_bytes: int = 4,
    relations: list[dict] | None = None,
    dense_map_budget_bytes: int = 64 * 1024 * 1024,
    num_target_nodes: int | None = None,
    num_source_nodes: int | None = None,
) -> dict[str, int | bool]:
    if relations is None:
        if num_relations is None or active_source_count is None or train_train_edges is None:
            raise ValueError("legacy dry-run mode requires num_relations, active_source_count, and train_train_edges")
        demand_cache_bytes = num_train_targets * num_relations * feature_dim * dtype_bytes
        edge_slice_cache_bytes = train_train_edges * (4 + 4 + 4)
        active_source_feature_bytes = active_source_count * feature_dim * dtype_bytes
        target_nodes = num_target_nodes if num_target_nodes is not None else num_train_targets
        return {
            "demand_cache_bytes": int(demand_cache_bytes),
            "edge_slice_cache_bytes": int(edge_slice_cache_bytes),
            "active_source_feature_bytes": int(active_source_feature_bytes),
            "expected_full_edge_scans": int(2 * num_relations),
            "total_expected_full_edge_scans": int(2 * num_relations),
            "peak_ram_estimate_bytes": int(demand_cache_bytes + edge_slice_cache_bytes + active_source_feature_bytes),
            "disk_spill_estimate_bytes": int(edge_slice_cache_bytes),
            "disk_spill_used": bool(edge_slice_cache_bytes > 2**31),
            "id_index_mode": select_id_index_mode(
                num_nodes=target_nodes,
                dense_map_budget_bytes=dense_map_budget_bytes,
            ),
        }

    relation_payload: dict[str, dict[str, int | str | bool]] = {}
    total_demand_cache_bytes = 0
    total_edge_slice_cache_bytes = 0
    total_active_source_feature_bytes = 0
    total_peak_ram_bytes = 0
    total_disk_spill_bytes = 0
    for idx, relation in enumerate(relations):
        name = str(relation.get("name", f"relation_{idx}"))
        source_is_target = bool(relation.get("source_is_target", False))
        active_sources = int(relation.get("num_active_sources", 0))
        train_train = int(relation.get("num_train_train_edges", 0)) if source_is_target else 0
        relation_target_nodes = int(relation.get("num_target_nodes", num_target_nodes or num_train_targets))
        relation_source_nodes = int(relation.get("num_source_nodes", num_source_nodes or active_sources))

        demand_cache_bytes = int(num_train_targets * feature_dim * dtype_bytes)
        edge_slice_cache_bytes = int(train_train * (4 + 4 + 4))
        active_source_feature_bytes = int(active_sources * feature_dim * dtype_bytes)
        dst_mode = select_id_index_mode(
            num_nodes=relation_target_nodes,
            dense_map_budget_bytes=dense_map_budget_bytes,
        )
        src_mode = (
            select_id_index_mode(
                num_nodes=relation_source_nodes,
                dense_map_budget_bytes=dense_map_budget_bytes,
            )
            if source_is_target
            else None
        )
        dst_index_bytes = int(relation_target_nodes * 4) if dst_mode == "dense_int32" else int(num_train_targets * 16)
        src_index_bytes = (
            int(relation_source_nodes * 4) if src_mode == "dense_int32" else int(num_train_targets * 16)
        ) if src_mode is not None else 0
        estimated_peak_ram_bytes = (
            demand_cache_bytes
            + edge_slice_cache_bytes
            + active_source_feature_bytes
            + dst_index_bytes
            + src_index_bytes
        )
        disk_spill_used = edge_slice_cache_bytes > 2**31
        relation_payload[name] = {
            "num_edges": int(relation.get("num_edges", 0)),
            "num_train_target_incident_edges": int(relation.get("num_train_target_incident_edges", 0)),
            "num_train_train_edges": train_train,
            "num_active_sources": active_sources,
            "source_is_target": source_is_target,
            "demand_cache_bytes": demand_cache_bytes,
            "edge_slice_cache_edges": train_train,
            "edge_slice_cache_bytes": edge_slice_cache_bytes,
            "edge_slice_dtype": "int32,int32,float32" if source_is_target else "none",
            "active_source_feature_bytes": active_source_feature_bytes,
            "id_index_mode": dst_mode,
            "dst_id_index_mode": dst_mode,
            "src_train_id_index_mode": src_mode or "none",
            "dst_id_index_bytes": dst_index_bytes,
            "src_train_id_index_bytes": src_index_bytes,
            "estimated_peak_ram_bytes": int(estimated_peak_ram_bytes),
            "estimated_disk_spill_bytes": int(edge_slice_cache_bytes),
            "disk_spill_used": bool(disk_spill_used),
            "expected_full_edge_scans": 2,
        }
        total_demand_cache_bytes += demand_cache_bytes
        total_edge_slice_cache_bytes += edge_slice_cache_bytes
        total_active_source_feature_bytes += active_source_feature_bytes
        total_peak_ram_bytes += estimated_peak_ram_bytes
        total_disk_spill_bytes += edge_slice_cache_bytes

    total_scans = 2 * len(relations)
    return {
        "demand_cache_bytes": int(total_demand_cache_bytes),
        "edge_slice_cache_bytes": int(total_edge_slice_cache_bytes),
        "active_source_feature_bytes": int(total_active_source_feature_bytes),
        "expected_full_edge_scans": int(total_scans),
        "total_expected_full_edge_scans": int(total_scans),
        "peak_ram_estimate_bytes": int(total_peak_ram_bytes),
        "disk_spill_estimate_bytes": int(total_disk_spill_bytes),
        "disk_spill_used": bool(total_edge_slice_cache_bytes > 2**31),
        "relations": relation_payload,
    }


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
    dense_map_budget_bytes: int = 64 * 1024 * 1024,
) -> RelationDemandCache:
    """Two-pass train-target-only streaming demand cache."""

    validate_train_target_only_cache(
        num_target_nodes=num_target_nodes,
        train_target_ids=train_target_ids,
        cache_all_targets=train_target_ids.numel() >= num_target_nodes,
        debug_allow_all_node_cache=debug_allow_all_node_cache,
    )
    train_target_ids = train_target_ids.to(torch.long).cpu()
    dst_to_pos = IdIndex.build(
        train_target_ids,
        num_nodes=num_target_nodes,
        dense_map_budget_bytes=dense_map_budget_bytes,
    )
    src_train_to_pos = (
        IdIndex.build(
            train_target_ids,
            num_nodes=num_source_nodes,
            dense_map_budget_bytes=dense_map_budget_bytes,
        )
        if source_is_target
        else None
    )

    degree_weight_sum = torch.zeros(train_target_ids.numel(), dtype=torch.float32)
    active_mask = torch.zeros(num_source_nodes, dtype=torch.bool)
    full_edge_scans = 0

    build_start = perf_counter()
    for chunk in edge_stream_factory():
        dst_pos = dst_to_pos.lookup(chunk.dst.cpu())
        mask = dst_pos >= 0
        if bool(mask.any()):
            degree_weight_sum.index_add_(0, dst_pos[mask], chunk.weight[mask].cpu())
            active_mask[chunk.src[mask].cpu()] = True
    full_edge_scans += 1
    cache_build_time = perf_counter() - build_start

    demand = torch.zeros(train_target_ids.numel(), feature_dim, dtype=torch.float32)
    edge_dst: list[torch.Tensor] = []
    edge_src: list[torch.Tensor] = []
    edge_alpha: list[torch.Tensor] = []

    aggregation_start = perf_counter()
    for chunk in edge_stream_factory():
        dst_pos = dst_to_pos.lookup(chunk.dst.cpu())
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
            selected_src_pos = src_train_to_pos.lookup(selected_src)
            slice_mask = selected_src_pos >= 0
            if bool(slice_mask.any()):
                edge_dst.append(selected_dst[slice_mask].to(torch.int32))
                edge_src.append(selected_src_pos[slice_mask].to(torch.int32))
                edge_alpha.append(alpha[slice_mask].to(torch.float32))
    full_edge_scans += 1
    cache_aggregation_time = perf_counter() - aggregation_start

    edge_slice = None
    if cache_edge_slice and edge_alpha:
        edge_slice = EdgeSliceCache(
            dst_train_pos=torch.cat(edge_dst),
            src_train_pos=torch.cat(edge_src),
            alpha=torch.cat(edge_alpha),
        )
    active_sources = torch.nonzero(active_mask, as_tuple=False).flatten().to(torch.long)
    edge_slice_cache_edges = 0 if edge_slice is None else edge_slice.num_edges
    edge_slice_cache_bytes = 0 if edge_slice is None else edge_slice.nbytes
    stats = RelationCacheStats(
        full_edge_scans=full_edge_scans,
        active_source_count=int(active_sources.numel()),
        edge_slice_cache_bytes=edge_slice_cache_bytes,
        edge_slice_cache_edges=edge_slice_cache_edges,
        edge_slice_dtype="none" if edge_slice is None else edge_slice.dtype_summary,
        cache_build_time=float(cache_build_time),
        cache_aggregation_time=float(cache_aggregation_time),
        disk_spill_used=bool(edge_slice_cache_bytes > 2**31),
        dst_id_index_mode=dst_to_pos.mode,
        src_train_id_index_mode=None if src_train_to_pos is None else src_train_to_pos.mode,
        dst_id_index_bytes=dst_to_pos.storage_nbytes,
        src_train_id_index_bytes=0 if src_train_to_pos is None else src_train_to_pos.storage_nbytes,
    )
    return RelationDemandCache(
        demand=demand,
        degree_weight_sum=degree_weight_sum,
        active_sources=active_sources,
        edge_slice=edge_slice,
        stats=stats,
    )
