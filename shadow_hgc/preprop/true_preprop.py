from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.block_stats import BlockStandardizer
from shadow_hgc.features.projection import fixed_random_projection
from shadow_hgc.preprop.chunked_spmm import chunked_destination_row_spmm
from shadow_hgc.preprop.manifest import PrepropBlockMeta, PrepropManifest
from shadow_hgc.preprop.memmap_store import write_tensor_memmap


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _target_features(x_provider, target_type: str) -> torch.Tensor:
    if isinstance(x_provider, Mapping):
        if target_type in x_provider:
            return x_provider[target_type].to(torch.float32)
        if "x" in x_provider:
            return x_provider["x"].to(torch.float32)
    raise ValueError(f"x_provider must contain target features for {target_type}")


def _train_rows(x_provider, num_rows: int) -> torch.Tensor:
    if isinstance(x_provider, Mapping) and "train_rows" in x_provider:
        return torch.as_tensor(x_provider["train_rows"], dtype=torch.long)
    return torch.arange(int(num_rows), dtype=torch.long)


def _feature_for_type(x_provider, node_type: str, fallback: torch.Tensor) -> torch.Tensor:
    if isinstance(x_provider, Mapping) and node_type in x_provider:
        return x_provider[node_type].to(torch.float32)
    return fallback


def _relations_mapping(relations) -> dict[DirectedRelation, torch.Tensor]:
    if isinstance(relations, Mapping):
        return {rel: edge.to(torch.long) for rel, edge in relations.items()}
    out: dict[DirectedRelation, torch.Tensor] = {}
    for item in relations:
        if isinstance(item, tuple) and len(item) == 2:
            rel, edge = item
            out[rel] = edge.to(torch.long)
        else:
            raise ValueError("relations must be a mapping or iterable of (DirectedRelation, edge_index)")
    return out


def _project(x: torch.Tensor, *, feature_dim: int, seed: int) -> torch.Tensor:
    out = x.to(torch.float32)
    if int(feature_dim) > 0 and int(out.shape[1]) > int(feature_dim):
        return fixed_random_projection(out, out_dim=int(feature_dim), seed=int(seed)).to(torch.float32)
    return out


def _relation_filter(
    block_name: str,
    *,
    target_type: str,
    relation_map: dict[DirectedRelation, torch.Tensor],
) -> list[DirectedRelation]:
    suffix = ""
    for prefix in ("X1_", "X2_"):
        if block_name.startswith(prefix):
            suffix = block_name[len(prefix) :]
    candidates = [rel for rel in relation_map if rel.source_type == target_type and rel.destination_type == target_type]
    if suffix:
        candidates = [rel for rel in candidates if rel.relation_name == suffix or str(rel).replace("--", "_").replace("-->", "_") == suffix]
    return sorted(candidates)


def _sum_relation_spmm(
    *,
    relations: list[DirectedRelation],
    relation_map: dict[DirectedRelation, torch.Tensor],
    source_features: torch.Tensor,
    num_dst_nodes: int,
    edge_chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    out = torch.zeros(num_dst_nodes, int(source_features.shape[1]), dtype=torch.float32)
    full_edge_scans = 0
    max_chunk = 0
    for relation in relations:
        result = chunked_destination_row_spmm(
            edge_index=relation_map[relation],
            source_features=source_features,
            num_dst_nodes=num_dst_nodes,
            edge_chunk_size=edge_chunk_size,
        )
        out += result.block
        full_edge_scans += int(result.diagnostics.get("full_edge_scans", 1))
        max_chunk = max(max_chunk, int(result.diagnostics.get("max_edge_chunk_size", 0)))
    return out, {
        "normalization": "destination_row",
        "edge_scans": int(full_edge_scans),
        "full_edge_scans": int(full_edge_scans),
        "max_edge_chunk_size": int(max_chunk),
        "uses_e_by_d_materialization": False,
        "materialized_full_e_by_d": False,
    }


def _typed_demand_block(
    *,
    x_provider,
    fallback_target_features: torch.Tensor,
    target_type: str,
    relation_map: dict[DirectedRelation, torch.Tensor],
    feature_dim: int,
    edge_chunk_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict, list[str]]:
    incoming = [rel for rel in relation_map if rel.destination_type == target_type and rel.source_type != target_type]
    pieces = []
    scans = 0
    max_chunk = 0
    for offset, relation in enumerate(sorted(incoming)):
        if isinstance(x_provider, Mapping) and relation.source_type in x_provider:
            raw_source = x_provider[relation.source_type].to(torch.float32)
        else:
            edge_index = relation_map[relation]
            num_source = int(edge_index[0].max().item()) + 1 if edge_index.numel() else 0
            raw_source = torch.zeros(num_source, int(fallback_target_features.shape[1]), dtype=torch.float32)
        source = _project(raw_source, feature_dim=feature_dim, seed=seed + offset + 1)
        result = chunked_destination_row_spmm(
            edge_index=relation_map[relation],
            source_features=source,
            num_dst_nodes=int(fallback_target_features.shape[0]),
            edge_chunk_size=edge_chunk_size,
        )
        pieces.append(result.block)
        scans += int(result.diagnostics.get("full_edge_scans", 1))
        max_chunk = max(max_chunk, int(result.diagnostics.get("max_edge_chunk_size", 0)))
    if not pieces:
        return torch.zeros(int(fallback_target_features.shape[0]), 1, dtype=torch.float32), {"normalization": "none", "full_edge_scans": 0}, []
    block = torch.cat(pieces, dim=1)
    if int(feature_dim) > 0 and int(block.shape[1]) > int(feature_dim):
        block = fixed_random_projection(block, out_dim=int(feature_dim), seed=int(seed) + 97).to(torch.float32)
    return block, {
        "normalization": "destination_row",
        "edge_scans": int(scans),
        "full_edge_scans": int(scans),
        "max_edge_chunk_size": int(max_chunk),
        "uses_e_by_d_materialization": False,
        "materialized_full_e_by_d": False,
    }, [str(rel) for rel in sorted(incoming)]


def _structure_block(
    *,
    relation_map: dict[DirectedRelation, torch.Tensor],
    target_type: str,
    num_rows: int,
) -> tuple[torch.Tensor, dict, list[str]]:
    pieces = []
    rels = []
    for relation in sorted(relation_map):
        if relation.destination_type != target_type:
            continue
        deg = torch.zeros(num_rows, dtype=torch.float32)
        edge_index = relation_map[relation]
        if edge_index.numel():
            deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.float32))
        pieces.append(torch.stack([torch.log1p(deg), (deg == 0).to(torch.float32)], dim=1))
        rels.append(str(relation))
    block = torch.cat(pieces, dim=1) if pieces else torch.zeros(num_rows, 0, dtype=torch.float32)
    return block, {
        "normalization": "none",
        "edge_scans": int(len(rels)),
        "full_edge_scans": int(len(rels)),
        "uses_e_by_d_materialization": False,
        "materialized_full_e_by_d": False,
    }, rels


def _write_stats(output: Path, block_name: str, block: torch.Tensor, train_rows: torch.Tensor) -> None:
    stats = BlockStandardizer.fit(block.to(torch.float32), train_rows=train_rows, block_name=block_name).freeze()
    (output / f"block_{block_name}_stats.json").write_text(json.dumps(stats.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def compute_preprop_blocks(
    dataset_name: str,
    target_type: str,
    x_provider,
    relations,
    output_dir: str,
    blocks: list[str],
    feature_dim: int = 128,
    dtype: str = "float16",
    edge_chunk_size: int = 2_000_000,
    dst_chunk_size: int = 200_000,
    max_ram_gb: float = 24.0,
    force_memmap: bool = True,
    seed: int = 42,
) -> PrepropManifest:
    del max_ram_gb
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    relation_map = _relations_mapping(relations)
    x0 = _project(_target_features(x_provider, target_type), feature_dim=int(feature_dim), seed=int(seed))
    train_rows = _train_rows(x_provider, int(x0.shape[0]))
    computed: dict[str, torch.Tensor] = {"X0": x0}
    metas: list[PrepropBlockMeta] = []
    total_bytes = 0
    full_edge_scans = 0

    def materialize(name: str, kind: str, block: torch.Tensor, diagnostics: dict, source_relations: list[str]) -> None:
        nonlocal total_bytes, full_edge_scans
        if force_memmap:
            rel_path = f"block_{name}.memmap"
            info = write_tensor_memmap(output / rel_path, block, dtype=dtype)
            disk_bytes = int(info["disk_bytes"])
            shape = [int(v) for v in info["shape"]]
            out_dtype = str(info["dtype"])
        else:
            rel_path = f"block_{name}.pt"
            torch.save(block.to(torch.float32), output / rel_path)
            disk_bytes = int(block.numel() * block.element_size())
            shape = [int(v) for v in block.shape]
            out_dtype = "float32"
        _write_stats(output, name, block, train_rows)
        scans = int(diagnostics.get("full_edge_scans", diagnostics.get("edge_scans", 0)))
        total_bytes += disk_bytes
        full_edge_scans += scans
        metas.append(
            PrepropBlockMeta(
                name=name,
                kind=kind,
                shape=shape,
                dtype=out_dtype,
                path=rel_path,
                source_relations=source_relations,
                normalization=str(diagnostics.get("normalization", "none")),
                stats_fit_source="train_target_rows",
                uses_logits=False,
                uses_teacher_logits=False,
                uses_kd=False,
                uses_diffusion_legacy=False,
                uses_dense_p2=False,
                uses_e_by_d_materialization=bool(diagnostics.get("uses_e_by_d_materialization", False)),
                uses_bounded_edges=False,
                edge_scans=scans,
                cache_bytes=disk_bytes,
                stats_fit_scope="train_target_rows",
                spec_hash=_hash_payload({"dataset": dataset_name, "name": name, "rels": source_relations, "seed": seed}),
                diagnostics=diagnostics,
            )
        )

    for requested in blocks:
        name = str(requested)
        if name == "X0":
            materialize("X0", "self", x0, {"normalization": "none", "full_edge_scans": 0}, [])
            continue
        if name.startswith("X1"):
            rels = _relation_filter(name, target_type=target_type, relation_map=relation_map)
            block, diag = _sum_relation_spmm(
                relations=rels,
                relation_map=relation_map,
                source_features=x0,
                num_dst_nodes=int(x0.shape[0]),
                edge_chunk_size=edge_chunk_size,
            )
            computed[name] = block
            if name == "X1":
                computed["X1"] = block
            materialize(name, "preprop_one_hop", block, diag, [str(rel) for rel in rels])
            continue
        if name.startswith("X2"):
            x1_name = "X1" if name == "X2" else "X1_" + name[len("X2_") :]
            if x1_name not in computed:
                rels_for_x1 = _relation_filter(x1_name, target_type=target_type, relation_map=relation_map)
                computed[x1_name], _ = _sum_relation_spmm(
                    relations=rels_for_x1,
                    relation_map=relation_map,
                    source_features=x0,
                    num_dst_nodes=int(x0.shape[0]),
                    edge_chunk_size=edge_chunk_size,
                )
            rels = _relation_filter(name, target_type=target_type, relation_map=relation_map)
            block, diag = _sum_relation_spmm(
                relations=rels,
                relation_map=relation_map,
                source_features=computed[x1_name],
                num_dst_nodes=int(x0.shape[0]),
                edge_chunk_size=edge_chunk_size,
            )
            computed[name] = block
            materialize(name, "preprop_two_hop", block, diag, [str(rel) for rel in rels])
            continue
        if name == "Xres":
            if "X1" not in computed:
                rels = _relation_filter("X1", target_type=target_type, relation_map=relation_map)
                computed["X1"], _ = _sum_relation_spmm(
                    relations=rels,
                    relation_map=relation_map,
                    source_features=x0,
                    num_dst_nodes=int(x0.shape[0]),
                    edge_chunk_size=edge_chunk_size,
                )
            block = x0 - computed["X1"]
            materialize("Xres", "residual", block, {"normalization": "none", "full_edge_scans": 0}, [])
            continue
        if name == "typed_demand":
            block, diag, source_relations = _typed_demand_block(
                x_provider=x_provider,
                fallback_target_features=x0,
                target_type=target_type,
                relation_map=relation_map,
                feature_dim=int(feature_dim),
                edge_chunk_size=edge_chunk_size,
                seed=int(seed),
            )
            materialize("typed_demand", "typed_demand", block, diag, source_relations)
            continue
        if name in {"structure", "LAD/SCAP", "lad_scap", "metapath"}:
            if name == "structure":
                block, diag, source_relations = _structure_block(relation_map=relation_map, target_type=target_type, num_rows=int(x0.shape[0]))
                materialize("structure", "structure", block, diag, source_relations)
            else:
                block = torch.zeros(int(x0.shape[0]), 1, dtype=torch.float32)
                materialize(name.replace("/", "_").lower(), name, block, {"normalization": "none", "full_edge_scans": 0}, [])
            continue
        raise ValueError(f"unsupported T2.1 preprop block: {name}")

    manifest = PrepropManifest(
        dataset=str(dataset_name),
        target_type=str(target_type),
        seed=int(seed),
        blocks=metas,
        total_cache_bytes=int(total_bytes),
        peak_cpu_ram_gb=current_cpu_ram_bytes() / (1024**3),
        peak_gpu_ram_gb=current_gpu_ram_bytes() / (1024**3),
        full_edge_scans=int(full_edge_scans),
        feature_hash=_hash_payload(tuple(x0.shape)),
        split_hash=_hash_payload(train_rows.tolist()),
        edge_chunk_size=int(edge_chunk_size),
        dst_chunk_size=int(dst_chunk_size),
        block_dim=int(feature_dim),
        uses_memmap=bool(force_memmap),
        uses_logits_as_input=False,
        uses_teacher_logits=False,
        uses_kd=False,
        uses_diffusion_legacy=False,
        uses_e_by_d_materialization=any(block.uses_e_by_d_materialization for block in metas),
        uses_dense_p2=False,
        uses_bounded_edges=False,
        wall_time_s=float(time.perf_counter() - started),
    )
    manifest.write(output)
    return manifest
