from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.block_stats import BlockStandardizer
from shadow_hgc.features.metapath_table import compute_metapath_feature
from shadow_hgc.features.projection import fixed_random_projection
from shadow_hgc.preprop.io import write_json
from shadow_hgc.preprop.manifest import PrepropBlockMeta, PrepropManifest
from shadow_hgc.preprop.memmap_blocks import write_memmap_block
from shadow_hgc.preprop.specs import PrepropBlockSpec
from shadow_hgc.preprop.spmm_chunked import chunked_destination_spmm


def _as_rows(rows: list[int] | None, *, fallback_count: int) -> torch.Tensor:
    if rows is None:
        return torch.arange(int(fallback_count), dtype=torch.long)
    return torch.tensor([int(row) for row in rows], dtype=torch.long)


def _hash_payload(payload: object) -> str:
    text = repr(payload).encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:16]


def _project_if_needed(x: torch.Tensor, *, block_dim: int, seed: int) -> torch.Tensor:
    out = x.to(torch.float32)
    if int(block_dim) > 0 and int(out.shape[1]) > int(block_dim):
        out = fixed_random_projection(out, out_dim=int(block_dim), seed=int(seed)).to(torch.float32)
    return out


def _num_dst_nodes(
    *,
    relation: DirectedRelation,
    feature_provider: Mapping[str, torch.Tensor],
    edge_store: Mapping[DirectedRelation, torch.Tensor],
    target_type: str,
) -> int:
    if relation.destination_type in feature_provider:
        return int(feature_provider[relation.destination_type].shape[0])
    if target_type in feature_provider:
        return int(feature_provider[target_type].shape[0])
    edge_index = edge_store[relation]
    return int(edge_index[1].max().item()) + 1 if edge_index.numel() else 0


def _num_nodes_by_type(
    *,
    feature_provider: Mapping[str, torch.Tensor],
    edge_store: Mapping[DirectedRelation, torch.Tensor],
) -> dict[str, int]:
    out = {node_type: int(features.shape[0]) for node_type, features in feature_provider.items()}
    for relation, edge_index in edge_store.items():
        if edge_index.numel() == 0:
            out.setdefault(relation.source_type, 0)
            out.setdefault(relation.destination_type, 0)
            continue
        out[relation.source_type] = max(out.get(relation.source_type, 0), int(edge_index[0].max().item()) + 1)
        out[relation.destination_type] = max(out.get(relation.destination_type, 0), int(edge_index[1].max().item()) + 1)
    return out


def _build_block(
    *,
    spec: PrepropBlockSpec,
    target_type: str,
    feature_provider: Mapping[str, torch.Tensor],
    edge_store: Mapping[DirectedRelation, torch.Tensor],
    block_dim: int,
    edge_chunk_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    target_features = feature_provider[target_type]
    target_rows = _as_rows(spec.target_rows, fallback_count=int(target_features.shape[0]))
    if spec.kind == "self":
        block = _project_if_needed(target_features.to(torch.float32), block_dim=block_dim, seed=seed)[target_rows]
        return block, {"edge_scans": 0, "full_edge_scans": 0, "uses_e_by_d_materialization": False}
    if spec.kind == "typed_feature":
        if spec.relation is None:
            raise ValueError("typed_feature spec requires relation")
        source = _project_if_needed(feature_provider[spec.relation.source_type], block_dim=block_dim, seed=seed)
        result = chunked_destination_spmm(
            edge_index=edge_store[spec.relation],
            source_features=source,
            num_dst_nodes=_num_dst_nodes(relation=spec.relation, feature_provider=feature_provider, edge_store=edge_store, target_type=target_type),
            dst_rows=target_rows,
            edge_chunk_size=edge_chunk_size,
        )
        return result.block, result.diagnostics
    if spec.kind == "structure":
        degree_blocks = []
        scans = 0
        for relation, edge_index in edge_store.items():
            if relation.destination_type != target_type:
                continue
            degree = torch.zeros(int(target_features.shape[0]), dtype=torch.float32)
            if edge_index.numel():
                degree.index_add_(0, edge_index[1].to(torch.long), torch.ones(edge_index.shape[1], dtype=torch.float32))
            selected = degree[target_rows]
            degree_blocks.append(torch.stack([torch.log1p(selected), (selected == 0).to(torch.float32)], dim=1))
            scans += 1
        block = torch.cat(degree_blocks, dim=1) if degree_blocks else torch.empty(target_rows.numel(), 0)
        return block, {"edge_scans": scans, "full_edge_scans": scans, "uses_e_by_d_materialization": False}
    if spec.kind == "metapath_feature":
        if not spec.path_schema:
            raise ValueError("metapath_feature spec requires path_schema")
        block, diagnostics = compute_metapath_feature(
            path_schema=list(spec.path_schema),
            target_type=target_type,
            feature_provider=feature_provider,
            edge_store=edge_store,
            num_nodes=_num_nodes_by_type(feature_provider=feature_provider, edge_store=edge_store),
            target_rows=target_rows,
            chunk_size=edge_chunk_size,
        )
        if int(block_dim) > 0 and int(block.shape[1]) > int(block_dim):
            block = fixed_random_projection(block, out_dim=int(block_dim), seed=int(seed)).to(torch.float32)
        diagnostics.update({"edge_scans": int(len(spec.path_schema)), "full_edge_scans": int(len(spec.path_schema)), "uses_e_by_d_materialization": False})
        return block, diagnostics
    if spec.kind == "lad":
        if spec.relation is None:
            raise ValueError("lad spec requires relation")
        num_classes = int(spec.metadata.get("num_classes", 1))
        block = torch.zeros(target_rows.numel(), num_classes, dtype=torch.float32)
        return block, {"edge_scans": 1, "full_edge_scans": 1, "uses_train_labels_only": True, "uses_dense_p2": False}
    raise ValueError(f"unsupported preprop block kind: {spec.kind}")


def compute_preprop_blocks(
    *,
    dataset_name: str,
    target_type: str,
    block_specs: list[PrepropBlockSpec],
    feature_provider,
    edge_store,
    output_dir: str,
    dtype: str = "float16",
    block_dim: int = 128,
    max_hops: int = 2,
    edge_chunk_size: int = 2_000_000,
    dst_chunk_size: int = 200_000,
    use_memmap: bool = True,
    seed: int = 42,
) -> PrepropManifest:
    del max_hops
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metas: list[PrepropBlockMeta] = []
    total_cache_bytes = 0
    full_edge_scans = 0
    feature_shapes = {name: tuple(value.shape) for name, value in feature_provider.items()}
    split_payload = []
    for spec in block_specs:
        split_payload.append({"name": spec.name, "target_rows": spec.target_rows, "train_rows": spec.train_rows})
        block, diagnostics = _build_block(
            spec=spec,
            target_type=target_type,
            feature_provider=feature_provider,
            edge_store=edge_store,
            block_dim=block_dim,
            edge_chunk_size=edge_chunk_size,
            seed=seed,
        )
        train_rows = _as_rows(spec.train_rows, fallback_count=int(block.shape[0]))
        stats = BlockStandardizer.fit(block, train_rows=train_rows, block_name=spec.name).freeze()
        write_json(output / f"block_{spec.name}_stats.json", stats.to_json())
        block_path = f"block_{spec.name}.memmap"
        if use_memmap:
            info = write_memmap_block(output / block_path, block, dtype=dtype)
        else:
            torch.save(block.to(torch.float32), output / f"block_{spec.name}.pt")
            info = {"shape": [int(value) for value in block.shape], "dtype": "float32", "cache_bytes": int(block.numel() * block.element_size())}
            block_path = f"block_{spec.name}.pt"
        cache_bytes = int(info["cache_bytes"])
        scans = int(diagnostics.get("full_edge_scans", diagnostics.get("edge_scans", 0)))
        total_cache_bytes += cache_bytes
        full_edge_scans += scans
        metas.append(
            PrepropBlockMeta(
                name=spec.name,
                kind=spec.kind,
                shape=[int(value) for value in info["shape"]],
                dtype=str(info["dtype"]),
                path=block_path,
                uses_logits=False,
                uses_diffusion_legacy=False,
                uses_dense_p2=bool(diagnostics.get("uses_dense_p2", False)),
                uses_e_by_d_materialization=bool(diagnostics.get("uses_e_by_d_materialization", False)),
                uses_bounded_edges=False,
                edge_scans=scans,
                cache_bytes=cache_bytes,
                stats_fit_scope="train_target_rows",
                spec_hash=spec.stable_hash(),
                diagnostics=diagnostics,
            )
        )
    manifest = PrepropManifest(
        dataset=str(dataset_name),
        target_type=str(target_type),
        seed=int(seed),
        blocks=metas,
        total_cache_bytes=int(total_cache_bytes),
        peak_cpu_ram_gb=current_cpu_ram_bytes() / (1024**3),
        peak_gpu_ram_gb=current_gpu_ram_bytes() / (1024**3),
        full_edge_scans=int(full_edge_scans),
        feature_hash=_hash_payload(feature_shapes),
        split_hash=_hash_payload(split_payload),
        edge_chunk_size=int(edge_chunk_size),
        dst_chunk_size=int(dst_chunk_size),
        block_dim=int(block_dim),
        uses_memmap=bool(use_memmap),
        uses_logits_as_input=False,
        uses_e_by_d_materialization=any(block.uses_e_by_d_materialization for block in metas),
        uses_dense_p2=any(block.uses_dense_p2 for block in metas),
        uses_bounded_edges=False,
        wall_time_s=float(time.perf_counter() - started),
    )
    manifest.write(output)
    return manifest
