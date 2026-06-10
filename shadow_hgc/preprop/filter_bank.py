from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _relations_from_graph_spec(graph_spec: Any) -> dict[DirectedRelation, torch.Tensor]:
    raw = graph_spec.get("relations", graph_spec.get("edge_index", {})) if isinstance(graph_spec, Mapping) else getattr(graph_spec, "edge_index")
    return {rel: edge.to(torch.long).cpu() for rel, edge in raw.items()}


def _target_type(graph_spec: Any) -> str:
    if isinstance(graph_spec, Mapping):
        return str(graph_spec.get("target_type", graph_spec.get("target_node_type", "target")))
    return str(getattr(graph_spec, "target_type"))


def _feature(feature_provider: Any, target_type: str) -> torch.Tensor:
    if isinstance(feature_provider, Mapping):
        if target_type in feature_provider:
            return feature_provider[target_type].to(torch.float32).cpu()
        if "x" in feature_provider:
            return feature_provider["x"].to(torch.float32).cpu()
    if torch.is_tensor(feature_provider):
        return feature_provider.to(torch.float32).cpu()
    raise ValueError(f"feature_provider must contain features for target type {target_type}")


def _project_if_needed(x: torch.Tensor, *, feature_dim: int, seed: int) -> torch.Tensor:
    out = x.to(torch.float32)
    if int(feature_dim) > 0 and int(out.shape[1]) > int(feature_dim):
        return fixed_random_projection(out, out_dim=int(feature_dim), seed=int(seed)).to(torch.float32)
    return out


def _relation_suffix(block_name: str) -> str:
    for prefix in ("X1_", "X2_", "Xres1_", "Xres2_", "Y1_", "Y2_", "Y3_"):
        if block_name.startswith(prefix):
            return block_name[len(prefix) :]
    return ""


def _relation_name_matches(relation: DirectedRelation, suffix: str) -> bool:
    if not suffix:
        return True
    return relation.relation_name == suffix or str(relation).replace("--", "_").replace("-->", "_") == suffix


def _target_target_relations(relation_map: Mapping[DirectedRelation, torch.Tensor], target_type: str, suffix: str = "") -> list[DirectedRelation]:
    return sorted(
        rel
        for rel in relation_map
        if rel.source_type == target_type and rel.destination_type == target_type and _relation_name_matches(rel, suffix)
    )


def _sum_spmm(
    *,
    relation_map: Mapping[DirectedRelation, torch.Tensor],
    relations: Sequence[DirectedRelation],
    source_features: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any], list[str]]:
    out = torch.zeros(int(num_nodes), int(source_features.shape[1]), dtype=torch.float32)
    scans = 0
    max_chunk = 0
    rel_names: list[str] = []
    for relation in relations:
        result = chunked_destination_row_spmm(
            edge_index=relation_map[relation],
            source_features=source_features,
            num_dst_nodes=int(num_nodes),
            edge_chunk_size=int(edge_chunk_size),
        )
        out += result.block
        scans += int(result.diagnostics.get("full_edge_scans", 1))
        max_chunk = max(max_chunk, int(result.diagnostics.get("max_edge_chunk_size", 0)))
        rel_names.append(str(relation))
    return out, {
        "normalization": "destination_row",
        "edge_scans": scans,
        "full_edge_scans": scans,
        "max_edge_chunk_size": max_chunk,
        "uses_e_by_d_materialization": False,
        "materialized_full_e_by_d": False,
    }, rel_names


def _label_stats(block: torch.Tensor, *, prefix: str) -> dict[str, torch.Tensor]:
    support = block.sum(dim=1, keepdim=True)
    probs = block / support.clamp_min(1e-12)
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=1, keepdim=True)
    max_affinity = block.max(dim=1, keepdim=True).values if block.numel() else torch.zeros(block.shape[0], 1)
    return {
        f"{prefix}_support": support.to(torch.float32),
        f"{prefix}_entropy": entropy.to(torch.float32),
        f"{prefix}_max_affinity": max_affinity.to(torch.float32),
    }


def compute_label_reuse_blocks(
    *,
    relation_blocks: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    train_target_ids: torch.Tensor,
    num_target_nodes: int,
    num_classes: int,
    steps: Sequence[int] = (1, 2, 3),
    prior_centering: bool = False,
    edge_chunk_size: int = 1_000_000,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    labels = labels.to(torch.long).cpu()
    train_rows = train_target_ids.to(torch.long).cpu()
    y0 = torch.zeros(int(num_target_nodes), int(num_classes), dtype=torch.float32)
    if train_rows.numel() > 0:
        y0[train_rows, labels[train_rows]] = 1.0
    train_hist = torch.bincount(labels[train_rows], minlength=int(num_classes)).to(torch.float32)
    train_prior = train_hist / train_hist.sum().clamp_min(1.0)
    blocks: dict[str, torch.Tensor] = {}
    support_stats: dict[str, dict[str, float]] = {}
    total_scans = 0
    for relation_name, edge_index in relation_blocks.items():
        current = y0
        for step in range(1, max(int(value) for value in steps) + 1):
            result = chunked_destination_row_spmm(
                edge_index=edge_index.to(torch.long),
                source_features=current,
                num_dst_nodes=int(num_target_nodes),
                edge_chunk_size=int(edge_chunk_size),
            )
            current = result.block
            total_scans += int(result.diagnostics.get("full_edge_scans", 1))
            if step in {int(value) for value in steps}:
                key = f"Y{step}_{relation_name}"
                blocks[key] = current.to(torch.float32)
                if prior_centering:
                    blocks[f"{key}_centered"] = current.to(torch.float32) - train_prior.view(1, -1)
                for stat_name, stat_block in _label_stats(current, prefix=key).items():
                    blocks[stat_name] = stat_block
                support_stats[key] = {
                    "support_mean": float(current.sum(dim=1).mean().item()) if current.numel() else 0.0,
                    "nonzero_row_ratio": float((current.sum(dim=1) > 0).to(torch.float32).mean().item()) if current.numel() else 0.0,
                }
    return blocks, {
        "uses_valid_labels": False,
        "uses_test_labels": False,
        "train_label_prior": [float(value) for value in train_prior.tolist()],
        "label_reuse_steps": [int(value) for value in steps],
        "label_block_dims": {name: int(block.shape[1]) for name, block in blocks.items()},
        "label_block_cache_bytes": sum(int(block.numel() * block.element_size()) for block in blocks.values()),
        "label_support_stats": support_stats,
        "full_edge_scans": int(total_scans),
        "uses_e_by_d_materialization": False,
    }


def _structure_blocks(
    *,
    relation_map: Mapping[DirectedRelation, torch.Tensor],
    target_type: str,
    num_nodes: int,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, dict[str, Any], list[str]]:
    pieces: list[torch.Tensor] = []
    rels: list[str] = []
    for relation in sorted(relation_map):
        if relation.destination_type != target_type:
            continue
        edge_index = relation_map[relation]
        dst_deg = torch.zeros(int(num_nodes), dtype=torch.float32)
        if edge_index.numel():
            dst_deg.index_add_(0, edge_index[1].to(torch.long), torch.ones(edge_index.shape[1], dtype=torch.float32))
        pieces.append(torch.stack([dst_deg, torch.log1p(dst_deg), (dst_deg == 0).to(torch.float32)], dim=1))
        rels.append(str(relation))
    labels = labels.to(torch.long).cpu()
    train_hist = torch.bincount(labels[train_rows.to(torch.long)], minlength=int(num_classes)).to(torch.float32)
    train_prior = train_hist / train_hist.sum().clamp_min(1.0)
    prior_block = train_prior.view(1, -1).repeat(int(num_nodes), 1)
    max_prior = prior_block.max(dim=1, keepdim=True).values
    entropy = -(prior_block.clamp_min(1e-12).log() * prior_block).sum(dim=1, keepdim=True)
    pieces.append(torch.cat([prior_block.sum(dim=1, keepdim=True), entropy, max_prior], dim=1))
    block = torch.cat(pieces, dim=1) if pieces else torch.zeros(int(num_nodes), 0, dtype=torch.float32)
    return block, {
        "normalization": "none",
        "edge_scans": len(rels),
        "full_edge_scans": len(rels),
        "uses_e_by_d_materialization": False,
        "train_label_prior": [float(value) for value in train_prior.tolist()],
    }, rels


def _write_block_stats(root: Path, all_stats: dict[str, Any]) -> None:
    (root / "block_stats.json").write_text(json.dumps(all_stats, indent=2, sort_keys=True), encoding="utf-8")


def _write_block_index(root: Path, blocks: list[PrepropBlockMeta]) -> None:
    with (root / "block_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_name", "kind", "shape", "dtype", "path", "cache_bytes", "edge_scans", "normalization"])
        writer.writeheader()
        for block in blocks:
            writer.writerow(
                {
                    "block_name": block.name,
                    "kind": block.kind,
                    "shape": "x".join(str(value) for value in block.shape),
                    "dtype": block.dtype,
                    "path": block.path,
                    "cache_bytes": block.cache_bytes,
                    "edge_scans": block.edge_scans,
                    "normalization": block.normalization,
                }
            )


def _t22_block_dict(block: PrepropBlockMeta) -> dict[str, Any]:
    out = block.to_dict()
    out["source_relation"] = block.source_relations[0] if block.source_relations else ""
    out["uses_logits_as_input"] = bool(block.uses_logits)
    out["is_train_label_block"] = block.kind in {"label_reuse", "label_stat"}
    out["is_feature_block"] = block.kind in {"self", "hop_block", "residual"}
    out["is_structure_block"] = block.kind == "structure"
    return out


def compute_preprop_filter_bank(
    *,
    dataset_name: str,
    graph_spec,
    feature_provider,
    target_node_ids,
    train_target_ids,
    labels,
    out_dir,
    blocks=("X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3"),
    feature_dim: int = 128,
    label_topk=None,
    dtype: str = "float16",
    edge_chunk_size: int = 1_000_000,
    dst_chunk_size: int = 200_000,
    normalization: str = "row",
    direction_policy: str = "dataset_default",
    fit_stats_on: str = "train_target_rows",
) -> PrepropManifest:
    del label_topk, direction_policy
    if normalization not in {"row", "destination_row"}:
        raise ValueError("T2.2 promoted preprop supports destination-row normalization only")
    if fit_stats_on != "train_target_rows":
        raise ValueError("block stats must be fit on train target rows")
    started = time.perf_counter()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_type = _target_type(graph_spec)
    relation_map = _relations_from_graph_spec(graph_spec)
    target_rows = torch.as_tensor(target_node_ids, dtype=torch.long).cpu()
    train_rows = torch.as_tensor(train_target_ids, dtype=torch.long).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    raw_x = _feature(feature_provider, target_type)
    x0_all = _project_if_needed(raw_x, feature_dim=int(feature_dim), seed=42)
    num_nodes = int(x0_all.shape[0])
    num_classes = int(labels[labels >= 0].max().item()) + 1 if labels.numel() else 0
    computed: dict[str, torch.Tensor] = {"X0": x0_all}
    metas: list[PrepropBlockMeta] = []
    stats_by_block: dict[str, Any] = {}
    total_cache_bytes = 0
    total_scans = 0

    def add_block(name: str, kind: str, block: torch.Tensor, diagnostics: dict[str, Any], source_relations: list[str]) -> None:
        nonlocal total_cache_bytes, total_scans
        block = block.to(torch.float32).cpu()
        path = f"block_{name}.memmap"
        info = write_tensor_memmap(root / path, block[target_rows], dtype=dtype)
        stats = BlockStandardizer.fit(block, train_rows=train_rows, block_name=name).freeze()
        stats_json = stats.to_json()
        stats_by_block[name] = stats_json
        (root / f"block_{name}_stats.json").write_text(json.dumps(stats_json, indent=2, sort_keys=True), encoding="utf-8")
        scans = int(diagnostics.get("full_edge_scans", diagnostics.get("edge_scans", 0)))
        total_scans += scans
        total_cache_bytes += int(info["disk_bytes"])
        metas.append(
            PrepropBlockMeta(
                name=name,
                kind=kind,
                shape=[int(value) for value in info["shape"]],
                dtype=str(info["dtype"]),
                path=path,
                source_relations=source_relations,
                normalization=str(diagnostics.get("normalization", "destination_row" if scans else "none")),
                stats_fit_source="train_target_rows",
                uses_logits=False,
                uses_teacher_logits=False,
                uses_kd=False,
                uses_diffusion_legacy=False,
                uses_dense_p2=False,
                uses_e_by_d_materialization=bool(diagnostics.get("uses_e_by_d_materialization", False)),
                uses_bounded_edges=False,
                edge_scans=scans,
                cache_bytes=int(info["disk_bytes"]),
                stats_fit_scope="train_target_rows",
                spec_hash=_hash_payload((dataset_name, name, source_relations, tuple(block.shape))),
                diagnostics={**diagnostics, "fit_stats_on": "train_target_rows"},
            )
        )

    def ensure_hop(name: str) -> torch.Tensor:
        if name in computed:
            return computed[name]
        suffix = _relation_suffix(name)
        rel_suffix = "" if suffix == "mix" else suffix
        rels = _target_target_relations(relation_map, target_type, suffix=rel_suffix)
        if name.startswith("X1"):
            source = x0_all
        elif name.startswith("X2"):
            source = ensure_hop("X1" + (f"_{rel_suffix}" if rel_suffix else ""))
        elif name.startswith("X3"):
            source = ensure_hop("X2" + (f"_{rel_suffix}" if rel_suffix else ""))
        elif name.startswith("X4"):
            source = ensure_hop("X3" + (f"_{rel_suffix}" if rel_suffix else ""))
        else:
            raise ValueError(f"unsupported hop block: {name}")
        block, diag, rel_names = _sum_spmm(
            relation_map=relation_map,
            relations=rels,
            source_features=source,
            num_nodes=num_nodes,
            edge_chunk_size=int(edge_chunk_size),
        )
        computed[name] = block
        computed[f"__diag__{name}"] = diag  # type: ignore[assignment]
        computed[f"__rels__{name}"] = rel_names  # type: ignore[assignment]
        return block

    label_relation_blocks = {
        rel.relation_name: relation_map[rel]
        for rel in _target_target_relations(relation_map, target_type)
    }
    label_blocks: dict[str, torch.Tensor] = {}
    label_diag: dict[str, Any] = {}

    for requested in blocks:
        name = str(requested)
        if name == "X0":
            add_block("X0", "self", x0_all, {"normalization": "none", "full_edge_scans": 0, "uses_e_by_d_materialization": False}, [])
        elif name.startswith(("X1", "X2", "X3", "X4")):
            block = ensure_hop(name)
            add_block(name, "hop_block", block, computed.get(f"__diag__{name}", {}), computed.get(f"__rels__{name}", []))  # type: ignore[arg-type]
        elif name.startswith("Xres1"):
            suffix = _relation_suffix(name)
            x1_name = "X1" + (f"_{suffix}" if suffix else "")
            block = x0_all - ensure_hop(x1_name)
            add_block(name, "residual", block, {"normalization": "none", "full_edge_scans": 0, "uses_e_by_d_materialization": False}, computed.get(f"__rels__{x1_name}", []))  # type: ignore[arg-type]
        elif name.startswith("Xres2"):
            suffix = _relation_suffix(name)
            x1_name = "X1" + (f"_{suffix}" if suffix else "")
            x2_name = "X2" + (f"_{suffix}" if suffix else "")
            block = ensure_hop(x1_name) - ensure_hop(x2_name)
            add_block(name, "residual", block, {"normalization": "none", "full_edge_scans": 0, "uses_e_by_d_materialization": False}, computed.get(f"__rels__{x2_name}", []))  # type: ignore[arg-type]
        elif name.startswith(("Y1", "Y2", "Y3")):
            if not label_blocks:
                label_blocks, label_diag = compute_label_reuse_blocks(
                    relation_blocks=label_relation_blocks,
                    labels=labels,
                    train_target_ids=train_rows,
                    num_target_nodes=num_nodes,
                    num_classes=num_classes,
                    steps=(1, 2, 3),
                    prior_centering=False,
                    edge_chunk_size=int(edge_chunk_size),
                )
            suffix = _relation_suffix(name)
            if suffix == "mix":
                step = name[1]
                candidates = [value for key, value in label_blocks.items() if key.startswith(f"Y{step}_") and "_support" not in key and "_entropy" not in key and "_max_affinity" not in key]
                block = sum(candidates) / max(1, len(candidates)) if candidates else torch.zeros(num_nodes, num_classes)
                rel_names = [str(rel) for rel in _target_target_relations(relation_map, target_type)]
            else:
                key = name if name in label_blocks else f"{name}_{suffix}" if suffix else name
                block = label_blocks.get(key)
                if block is None and not suffix and label_relation_blocks:
                    candidates = [
                        value
                        for key_name, value in label_blocks.items()
                        if key_name.startswith(f"{name}_")
                        and "_support" not in key_name
                        and "_entropy" not in key_name
                        and "_max_affinity" not in key_name
                    ]
                    block = sum(candidates) / max(1, len(candidates)) if candidates else None
                if block is None:
                    block = torch.zeros(num_nodes, num_classes)
                rel_names = [str(rel) for rel in _target_target_relations(relation_map, target_type, suffix=suffix)]
            add_block(name, "label_reuse", block, {"normalization": "destination_row", "full_edge_scans": 1, "uses_e_by_d_materialization": False, **label_diag}, rel_names)
        elif name == "structure":
            block, diag, rel_names = _structure_blocks(
                relation_map=relation_map,
                target_type=target_type,
                num_nodes=num_nodes,
                labels=labels,
                train_rows=train_rows,
                num_classes=num_classes,
            )
            add_block("structure", "structure", block, diag, rel_names)
        else:
            raise ValueError(f"unsupported T2.2 filter-bank block: {name}")

    manifest = PrepropManifest(
        dataset=str(dataset_name),
        target_type=target_type,
        seed=42,
        blocks=metas,
        total_cache_bytes=int(total_cache_bytes),
        peak_cpu_ram_gb=current_cpu_ram_bytes() / (1024**3),
        peak_gpu_ram_gb=current_gpu_ram_bytes() / (1024**3),
        full_edge_scans=int(total_scans),
        feature_hash=_hash_payload(tuple(x0_all.shape)),
        split_hash=_hash_payload(train_rows.tolist()),
        edge_chunk_size=int(edge_chunk_size),
        dst_chunk_size=int(dst_chunk_size),
        block_dim=int(feature_dim),
        uses_memmap=True,
        uses_logits_as_input=False,
        uses_teacher_logits=False,
        uses_kd=False,
        uses_diffusion_legacy=False,
        uses_e_by_d_materialization=any(block.uses_e_by_d_materialization for block in metas),
        uses_dense_p2=False,
        uses_bounded_edges=False,
        wall_time_s=float(time.perf_counter() - started),
    )
    manifest.write(root)
    t22_manifest = manifest.to_dict()
    t22_manifest["blocks"] = [_t22_block_dict(block) for block in metas]
    (root / "preprop_manifest.json").write_text(json.dumps(t22_manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_block_index(root, metas)
    _write_block_stats(root, stats_by_block)
    return manifest
