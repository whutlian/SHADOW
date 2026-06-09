from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrepropBlockMeta:
    name: str
    kind: str
    shape: list[int]
    dtype: str
    path: str
    source_relations: list[str] = field(default_factory=list)
    normalization: str = "none"
    stats_fit_source: str = "train_target_rows"
    uses_logits: bool = False
    uses_teacher_logits: bool = False
    uses_kd: bool = False
    uses_diffusion_legacy: bool = False
    uses_dense_p2: bool = False
    uses_e_by_d_materialization: bool = False
    uses_bounded_edges: bool = False
    edge_scans: int = 0
    cache_bytes: int = 0
    stats_fit_scope: str = "train_target_rows"
    spec_hash: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": [int(value) for value in self.shape],
            "dtype": self.dtype,
            "path": self.path,
            "source_relations": list(self.source_relations),
            "normalization": self.normalization,
            "stats_fit_source": self.stats_fit_source,
            "uses_logits": bool(self.uses_logits),
            "uses_teacher_logits": bool(self.uses_teacher_logits),
            "uses_kd": bool(self.uses_kd),
            "uses_diffusion_legacy": bool(self.uses_diffusion_legacy),
            "uses_dense_p2": bool(self.uses_dense_p2),
            "uses_e_by_d_materialization": bool(self.uses_e_by_d_materialization),
            "uses_bounded_edges": bool(self.uses_bounded_edges),
            "edge_scans": int(self.edge_scans),
            "cache_bytes": int(self.cache_bytes),
            "disk_bytes": int(self.cache_bytes),
            "stats_fit_scope": self.stats_fit_scope,
            "spec_hash": self.spec_hash,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class PrepropManifest:
    dataset: str
    target_type: str
    seed: int
    blocks: list[PrepropBlockMeta]
    total_cache_bytes: int
    peak_cpu_ram_gb: float
    peak_gpu_ram_gb: float
    full_edge_scans: int
    feature_hash: str
    split_hash: str
    edge_chunk_size: int
    dst_chunk_size: int
    block_dim: int
    uses_memmap: bool = True
    uses_logits_as_input: bool = False
    uses_teacher_logits: bool = False
    uses_kd: bool = False
    uses_diffusion_legacy: bool = False
    uses_e_by_d_materialization: bool = False
    uses_dense_p2: bool = False
    uses_bounded_edges: bool = False
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "target_type": self.target_type,
            "seed": int(self.seed),
            "blocks": [block.to_dict() for block in self.blocks],
            "total_cache_bytes": int(self.total_cache_bytes),
            "peak_cpu_ram_gb": float(self.peak_cpu_ram_gb),
            "peak_gpu_ram_gb": float(self.peak_gpu_ram_gb),
            "full_edge_scans": int(self.full_edge_scans),
            "feature_hash": self.feature_hash,
            "split_hash": self.split_hash,
            "edge_chunk_size": int(self.edge_chunk_size),
            "dst_chunk_size": int(self.dst_chunk_size),
            "block_dim": int(self.block_dim),
            "feature_dim": int(self.block_dim),
            "uses_memmap": bool(self.uses_memmap),
            "uses_logits_as_input": bool(self.uses_logits_as_input),
            "uses_teacher_logits": bool(self.uses_teacher_logits),
            "uses_kd": bool(self.uses_kd),
            "uses_diffusion_legacy": bool(self.uses_diffusion_legacy),
            "uses_e_by_d_materialization": bool(self.uses_e_by_d_materialization),
            "uses_dense_p2": bool(self.uses_dense_p2),
            "uses_bounded_edges": bool(self.uses_bounded_edges),
            "wall_time_s": float(self.wall_time_s),
        }

    def write(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


REQUIRED_T2_RESOURCE_FIELDS = [
    "full_edge_scans",
    "edge_chunk_size",
    "dst_chunk_size",
    "block_dim",
    "num_blocks",
    "cache_bytes",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "wall_time_s",
    "uses_memmap",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_logits_as_input",
    "uses_bounded_edges",
]


def validate_t2_resource_report(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_T2_RESOURCE_FIELDS if field not in row]
    forbidden = []
    for field in ["uses_e_by_d_materialization", "uses_dense_p2", "uses_logits_as_input", "uses_bounded_edges"]:
        if bool(row.get(field, False)):
            forbidden.append(field)
    return {
        "valid": not missing and not forbidden,
        "missing_fields": missing,
        "forbidden_flags": forbidden,
    }
