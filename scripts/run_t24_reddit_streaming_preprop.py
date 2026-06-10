from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.edge_stream import MemmapEdgeStream
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_manifest, prepare_reddit_raw_memmaps
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.preprop.manifest import PrepropBlockMeta, PrepropManifest
from shadow_hgc.preprop.memmap_blocks import numpy_dtype, torch_dtype
from shadow_hgc.preprop.memmap_store import write_tensor_memmap
from shadow_hgc.preprop.streaming_spmm import streaming_destination_row_spmm


FIELDS = [
    "dataset",
    "status",
    "reason",
    "blocks",
    "num_blocks",
    "num_nodes",
    "num_edges",
    "edge_limit",
    "feature_dim",
    "block_dim",
    "edge_chunk_size",
    "full_edge_scans",
    "cache_bytes",
    "wall_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "uses_memmap",
    "uses_processed_data_pt",
    "uses_e_by_d_materialization",
    "materialized_stacked_edge_index",
    "manifest_dir",
]


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _edge_factory(manifest: dict[str, Any], *, chunk_size: int, edge_limit: int | None):
    root = Path(manifest["memmap_root"])
    return lambda: MemmapEdgeStream(root / manifest["src_path"], root / manifest["dst_path"], chunk_size=chunk_size, edge_limit=edge_limit)


def _open_raw_memmap(path: Path, *, shape: tuple[int, ...], dtype: str) -> np.memmap:
    return np.memmap(path, mode="r", dtype=np.dtype(dtype), shape=shape)


def _source_getter(path: Path, *, shape: tuple[int, int], dtype: str):
    array = _open_raw_memmap(path, shape=shape, dtype=dtype)

    def get(ids: torch.Tensor) -> torch.Tensor:
        index = ids.detach().cpu().numpy().astype(np.int64, copy=False)
        values = np.asarray(array[index], dtype=np.float32)
        return torch.from_numpy(values)

    return get


def _write_stats(path: Path, *, name: str, shape: tuple[int, int], dtype: str, train_idx: np.ndarray) -> dict[str, Any]:
    array = _open_raw_memmap(path, shape=shape, dtype=dtype)
    scoped = np.asarray(array[train_idx.astype(np.int64, copy=False)], dtype=np.float32)
    tensor = torch.from_numpy(scoped)
    std = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    mean = tensor.mean(dim=0)
    payload = {
        "block_name": name,
        "fit_scope": "train_target_rows",
        "frozen": True,
        "eps": 1e-6,
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
        "norm_mean": float(torch.linalg.vector_norm(mean).item()),
        "std_mean": float(std.mean().item()) if std.numel() else 0.0,
        "fit_rows": [] if int(train_idx.size) > 1024 else [int(value) for value in train_idx.tolist()],
        "fit_rows_count": int(train_idx.size),
        "fit_rows_head": [int(value) for value in train_idx[:16].tolist()],
        "fit_rows_tail": [int(value) for value in train_idx[-16:].tolist()],
    }
    return payload


def _write_block_stats(root: Path, stats_by_block: dict[str, Any]) -> None:
    (root / "block_stats.json").write_text(json.dumps(stats_by_block, indent=2, sort_keys=True), encoding="utf-8")
    for name, stats in stats_by_block.items():
        (root / f"block_{name}_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


def _write_block_index(root: Path, metas: list[PrepropBlockMeta]) -> None:
    with (root / "block_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_name", "kind", "shape", "dtype", "path", "cache_bytes", "edge_scans", "normalization"])
        writer.writeheader()
        for meta in metas:
            writer.writerow(
                {
                    "block_name": meta.name,
                    "kind": meta.kind,
                    "shape": "x".join(str(value) for value in meta.shape),
                    "dtype": meta.dtype,
                    "path": meta.path,
                    "cache_bytes": meta.cache_bytes,
                    "edge_scans": meta.edge_scans,
                    "normalization": meta.normalization,
                }
            )


def _project_x0_to_memmap(
    *,
    source_feature_path: Path,
    source_shape: tuple[int, int],
    out_path: Path,
    out_dim: int,
    dtype: str,
    chunk_rows: int,
    seed: int,
) -> dict[str, Any]:
    source = np.load(source_feature_path, mmap_mode="r")
    if tuple(int(v) for v in source.shape) != source_shape:
        raise ValueError("source feature shape does not match manifest")
    np_dtype = numpy_dtype(dtype)
    out = np.memmap(out_path, mode="w+", dtype=np_dtype, shape=(source_shape[0], int(out_dim)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(source_shape[1], int(out_dim), generator=generator, dtype=torch.float32)
    projection = (projection / (source_shape[1] ** 0.5)).numpy()
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, int(source_shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(source_shape[0]))
        values = np.asarray(source[start:stop], dtype=np.float32) @ projection
        if np_dtype == np.dtype("float16"):
            values = values.astype(np.float16)
        out[start:stop] = values
    out.flush()
    return {"shape": [int(source_shape[0]), int(out_dim)], "dtype": np_dtype.name, "disk_bytes": int(out.size * np_dtype.itemsize)}


def _write_tensor_block(root: Path, name: str, tensor: torch.Tensor, *, dtype: str) -> dict[str, Any]:
    path = root / f"block_{name}.memmap"
    info = write_tensor_memmap(path, tensor, dtype=dtype)
    return {"path": path, "shape": [int(v) for v in info["shape"]], "dtype": str(info["dtype"]), "disk_bytes": int(info["disk_bytes"])}


def _write_residual_block(
    *,
    root: Path,
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    dtype: str,
    chunk_rows: int,
) -> dict[str, Any]:
    if tuple(left["shape"]) != tuple(right["shape"]):
        raise ValueError("residual block inputs must have the same shape")
    shape = tuple(int(v) for v in left["shape"])
    np_dtype = numpy_dtype(dtype)
    left_array = _open_raw_memmap(root / left["path"], shape=shape, dtype=left["dtype"])
    right_array = _open_raw_memmap(root / right["path"], shape=shape, dtype=right["dtype"])
    out_path = root / f"block_{name}.memmap"
    out = np.memmap(out_path, mode="w+", dtype=np_dtype, shape=shape)
    for start in range(0, int(shape[0]), max(1, int(chunk_rows))):
        stop = min(start + int(chunk_rows), int(shape[0]))
        values = np.asarray(left_array[start:stop], dtype=np.float32) - np.asarray(right_array[start:stop], dtype=np.float32)
        out[start:stop] = values.astype(np_dtype, copy=False)
    out.flush()
    return {"path": out_path, "shape": [int(v) for v in shape], "dtype": np_dtype.name, "disk_bytes": int(out.size * np_dtype.itemsize)}


def _structure_block(edge_stream_factory, *, num_nodes: int) -> tuple[torch.Tensor, dict[str, Any]]:
    started = time.perf_counter()
    degree = torch.zeros(int(num_nodes), dtype=torch.float32)
    max_chunk = 0
    num_edges = 0
    for chunk in edge_stream_factory():
        dst = chunk.dst.to(torch.long).cpu()
        weight = chunk.weight.to(torch.float32).cpu()
        max_chunk = max(max_chunk, int(dst.numel()))
        num_edges += int(dst.numel())
        if dst.numel() > 0:
            degree.index_add_(0, dst, weight)
    block = torch.stack([degree, torch.log1p(degree), (degree == 0).to(torch.float32)], dim=1)
    return block, {
        "normalization": "none",
        "full_edge_scans": 1,
        "edge_scans": 1,
        "num_edges": int(num_edges),
        "max_edge_chunk_size": int(max_chunk),
        "uses_e_by_d_materialization": False,
        "materialized_stacked_edge_index": False,
        "wall_time_s": float(time.perf_counter() - started),
    }


def _run_streaming_preprop(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[PrepropManifest, list[dict[str, Any]]]:
    started = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    memmap_root = Path(manifest["memmap_root"])
    train_idx = np.load(memmap_root / manifest["train_idx_path"], mmap_mode="r").astype(np.int64, copy=False)
    labels = np.load(memmap_root / manifest["label_path"], mmap_mode="r")
    num_nodes = int(manifest["num_nodes"])
    num_classes = int(manifest["num_classes"])
    requested = [str(block) for block in args.blocks]
    edge_limit = None if int(args.edge_limit) <= 0 else int(args.edge_limit)
    edge_factory = _edge_factory(manifest, chunk_size=int(args.edge_chunk_size), edge_limit=edge_limit)
    relation = "reddit_node--links-->reddit_node"

    block_infos: dict[str, dict[str, Any]] = {}
    metas: list[PrepropBlockMeta] = []
    stats_by_block: dict[str, Any] = {}
    total_cache_bytes = 0
    total_scans = 0
    rows: list[dict[str, Any]] = []

    def add_meta(name: str, kind: str, info: dict[str, Any], diagnostics: dict[str, Any], source_relations: list[str]) -> None:
        nonlocal total_cache_bytes, total_scans
        rel_path = Path(info["path"]).name
        info = {**info, "path": rel_path}
        block_infos[name] = info
        scans = int(diagnostics.get("full_edge_scans", diagnostics.get("edge_scans", 0)))
        total_scans += scans
        total_cache_bytes += int(info["disk_bytes"])
        stats = _write_stats(out / rel_path, name=name, shape=tuple(int(v) for v in info["shape"]), dtype=info["dtype"], train_idx=np.asarray(train_idx))
        stats_by_block[name] = stats
        metas.append(
            PrepropBlockMeta(
                name=name,
                kind=kind,
                shape=[int(v) for v in info["shape"]],
                dtype=str(info["dtype"]),
                path=rel_path,
                source_relations=source_relations,
                normalization=str(diagnostics.get("normalization", "destination_row" if scans else "none")),
                stats_fit_source="train_target_rows",
                uses_logits=False,
                uses_teacher_logits=False,
                uses_kd=False,
                uses_diffusion_legacy=False,
                uses_dense_p2=False,
                uses_e_by_d_materialization=False,
                uses_bounded_edges=False,
                edge_scans=scans,
                cache_bytes=int(info["disk_bytes"]),
                stats_fit_scope="train_target_rows",
                spec_hash=_hash_payload(("Reddit", name, tuple(info["shape"]), edge_limit)),
                diagnostics={
                    **diagnostics,
                    "fit_stats_on": "train_target_rows",
                    "edge_limit": "" if edge_limit is None else int(edge_limit),
                },
            )
        )
        rows.append({"block": name, "kind": kind, "shape": "x".join(str(v) for v in info["shape"]), "edge_scans": scans, "cache_bytes": int(info["disk_bytes"])})

    if "X0" in requested:
        x_info = _project_x0_to_memmap(
            source_feature_path=memmap_root / manifest["feature_path"],
            source_shape=(int(manifest["num_nodes"]), int(manifest["feature_dim"])),
            out_path=out / "block_X0.memmap",
            out_dim=int(args.feature_dim),
            dtype=args.dtype,
            chunk_rows=int(args.feature_chunk_rows),
            seed=int(args.seed),
        )
        x_info["path"] = out / "block_X0.memmap"
        add_meta("X0", "self", x_info, {"normalization": "none", "full_edge_scans": 0, "uses_e_by_d_materialization": False, "materialized_stacked_edge_index": False}, [])

    def ensure_x(name: str) -> dict[str, Any]:
        if name in block_infos:
            return block_infos[name]
        if name == "X1":
            source = ensure_x("X0")
        elif name == "X2":
            source = ensure_x("X1")
        elif name == "X3":
            source = ensure_x("X2")
        else:
            raise ValueError(f"unsupported X block: {name}")
        result = streaming_destination_row_spmm(
            edge_stream_factory=edge_factory,
            source_feature_getter=_source_getter(out / source["path"], shape=tuple(int(v) for v in source["shape"]), dtype=source["dtype"]),
            feature_dim=int(source["shape"][1]),
            num_dst_nodes=num_nodes,
            dst_rows=torch.arange(num_nodes, dtype=torch.long),
        )
        info = _write_tensor_block(out, name, result.block, dtype=args.dtype)
        add_meta(name, "hop_block", info, result.diagnostics, [relation])
        return block_infos[name]

    for name in requested:
        if name in {"X0", "structure"} or name.startswith("Y") or name.startswith("Xres"):
            continue
        if name in {"X1", "X2", "X3"}:
            ensure_x(name)
        else:
            raise ValueError(f"unsupported Reddit streaming block: {name}")

    if "Xres1" in requested:
        x0 = ensure_x("X0")
        x1 = ensure_x("X1")
        info = _write_residual_block(root=out, name="Xres1", left=x0, right=x1, dtype=args.dtype, chunk_rows=int(args.feature_chunk_rows))
        add_meta("Xres1", "residual", info, {"normalization": "none", "full_edge_scans": 0, "uses_e_by_d_materialization": False, "materialized_stacked_edge_index": False}, [relation])

    y_current: torch.Tensor | None = None
    y_steps = sorted({int(name[1]) for name in requested if name in {"Y1", "Y2", "Y3"}})
    if y_steps:
        y_current = torch.zeros(num_nodes, num_classes, dtype=torch.float32)
        train_tensor = torch.from_numpy(np.asarray(train_idx, dtype=np.int64).copy())
        train_labels = torch.from_numpy(np.asarray(labels[train_idx], dtype=np.int64).copy())
        y_current[train_tensor, train_labels] = 1.0
        for step in range(1, max(y_steps) + 1):
            result = streaming_destination_row_spmm(
                edge_stream_factory=edge_factory,
                source_feature_getter=lambda ids, current=y_current: current[ids.to(torch.long)],
                feature_dim=num_classes,
                num_dst_nodes=num_nodes,
                dst_rows=torch.arange(num_nodes, dtype=torch.long),
            )
            y_current = result.block
            if step in y_steps:
                name = f"Y{step}"
                info = _write_tensor_block(out, name, y_current, dtype=args.dtype)
                add_meta(name, "label_reuse", info, {**result.diagnostics, "uses_valid_labels": False, "uses_test_labels": False}, [relation])

    if "structure" in requested:
        block, diag = _structure_block(edge_factory, num_nodes=num_nodes)
        info = _write_tensor_block(out, "structure", block, dtype=args.dtype)
        add_meta("structure", "structure", info, diag, [relation])

    _write_block_stats(out, stats_by_block)
    _write_block_index(out, metas)
    preprop_manifest = PrepropManifest(
        dataset="Reddit",
        target_type="reddit_node",
        seed=int(args.seed),
        blocks=metas,
        total_cache_bytes=int(total_cache_bytes),
        peak_cpu_ram_gb=current_cpu_ram_bytes() / (1024**3),
        peak_gpu_ram_gb=current_gpu_ram_bytes() / (1024**3),
        full_edge_scans=int(total_scans),
        feature_hash=_hash_payload((manifest["feature_path"], manifest["feature_dim"], args.feature_dim)),
        split_hash=_hash_payload((int(train_idx.size), int(np.asarray(train_idx[:1]).sum()) if train_idx.size else 0)),
        edge_chunk_size=int(args.edge_chunk_size),
        dst_chunk_size=int(num_nodes),
        block_dim=int(args.feature_dim),
        uses_memmap=True,
        uses_logits_as_input=False,
        uses_teacher_logits=False,
        uses_kd=False,
        uses_diffusion_legacy=False,
        uses_e_by_d_materialization=False,
        uses_dense_p2=False,
        uses_bounded_edges=False,
        wall_time_s=float(time.perf_counter() - started),
    )
    preprop_manifest.write(out)
    (out / "preprop_manifest.json").write_text(json.dumps(preprop_manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return preprop_manifest, rows


def _dry_run_row(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    blocks = [str(block) for block in args.blocks]
    edge_scans = 0
    for block in blocks:
        if block in {"X1", "X2", "X3", "Y1", "Y2", "Y3"}:
            edge_scans += 2
        elif block == "structure":
            edge_scans += 1
    num_nodes = int(manifest["num_nodes"])
    num_classes = int(manifest["num_classes"])
    dtype_bytes = np.dtype(numpy_dtype(args.dtype)).itemsize
    cache_bytes = 0
    for block in blocks:
        if block.startswith("Y"):
            dim = num_classes
        elif block == "structure":
            dim = 3
        else:
            dim = int(args.feature_dim)
        cache_bytes += num_nodes * dim * dtype_bytes
    return {
        "dataset": "Reddit",
        "status": "dry_run",
        "reason": "streaming preprop estimate only",
        "blocks": ",".join(blocks),
        "num_blocks": len(blocks),
        "num_nodes": num_nodes,
        "num_edges": int(manifest["num_edges"]),
        "edge_limit": "" if int(args.edge_limit) <= 0 else int(args.edge_limit),
        "feature_dim": int(manifest["feature_dim"]),
        "block_dim": int(args.feature_dim),
        "edge_chunk_size": int(args.edge_chunk_size),
        "full_edge_scans": edge_scans,
        "cache_bytes": int(cache_bytes),
        "wall_time_s": 0,
        "peak_cpu_ram_gb": "",
        "peak_gpu_ram_gb": "",
        "uses_memmap": True,
        "uses_processed_data_pt": False,
        "uses_e_by_d_materialization": False,
        "materialized_stacked_edge_index": False,
        "manifest_dir": args.out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T24 Reddit streaming raw-memmap preprop smoke/full job.")
    parser.add_argument("--reddit-root", default="dataset/Reddit")
    parser.add_argument("--memmap-root", default="")
    parser.add_argument("--out-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--overwrite-memmap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--blocks", nargs="+", default=["X0", "X1", "structure"])
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--edge-limit", type=int, default=0)
    parser.add_argument("--feature-chunk-rows", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default="experiments/tables/t24_reddit_streaming_preprop_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_reddit_streaming_preprop_summary.md")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        memmap_root = Path(args.memmap_root) if args.memmap_root else Path(args.reddit_root) / "processed" / "raw_memmap"
        if args.prepare or not (memmap_root / "manifest.json").exists():
            manifest = prepare_reddit_raw_memmaps(
                args.reddit_root,
                out_dir=memmap_root,
                overwrite=bool(args.overwrite_memmap),
                feature_chunk_rows=int(args.feature_chunk_rows),
            )
        else:
            manifest = load_reddit_raw_memmap_manifest(memmap_root)
        if args.dry_run:
            row = _dry_run_row(args, manifest)
            block_rows: list[dict[str, Any]] = []
        else:
            preprop_manifest, block_rows = _run_streaming_preprop(args, manifest)
            row = {
                "dataset": "Reddit",
                "status": "completed_streaming_smoke" if int(args.edge_limit) > 0 else "completed_streaming_full",
                "reason": "streaming raw-memmap preprop completed without processed data.pt or stacked edge_index",
                "blocks": ",".join(args.blocks),
                "num_blocks": len(preprop_manifest.blocks),
                "num_nodes": manifest["num_nodes"],
                "num_edges": manifest["num_edges"],
                "edge_limit": "" if int(args.edge_limit) <= 0 else int(args.edge_limit),
                "feature_dim": manifest["feature_dim"],
                "block_dim": int(args.feature_dim),
                "edge_chunk_size": int(args.edge_chunk_size),
                "full_edge_scans": preprop_manifest.full_edge_scans,
                "cache_bytes": preprop_manifest.total_cache_bytes,
                "wall_time_s": preprop_manifest.wall_time_s,
                "peak_cpu_ram_gb": preprop_manifest.peak_cpu_ram_gb,
                "peak_gpu_ram_gb": preprop_manifest.peak_gpu_ram_gb,
                "uses_memmap": True,
                "uses_processed_data_pt": False,
                "uses_e_by_d_materialization": preprop_manifest.uses_e_by_d_materialization,
                "materialized_stacked_edge_index": False,
                "manifest_dir": str(args.out_dir),
            }
    except Exception as exc:
        row = {
            "dataset": "Reddit",
            "status": "blocked",
            "reason": f"{type(exc).__name__}: {exc}",
            "blocks": ",".join(args.blocks),
            "num_blocks": 0,
            "wall_time_s": float(time.perf_counter() - started),
            "uses_memmap": True,
            "uses_processed_data_pt": False,
            "uses_e_by_d_materialization": False,
            "materialized_stacked_edge_index": False,
            "manifest_dir": str(args.out_dir),
        }
        block_rows = []

    output = write_csv(args.csv, [row], FIELDS)
    block_lines = markdown_table(block_rows, ["block", "kind", "shape", "edge_scans", "cache_bytes"]) if block_rows else ["_No block rows for dry-run or blocked run._"]
    ensure_report(
        args.report,
        [
            "# T24 Reddit Streaming Preprop",
            "",
            *markdown_table([row], ["dataset", "status", "blocks", "edge_limit", "full_edge_scans", "cache_bytes", "peak_cpu_ram_gb", "reason"]),
            "",
            "## Blocks",
            "",
            *block_lines,
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": row["status"], "csv": str(output), "manifest_dir": row.get("manifest_dir", "")}, sort_keys=True))


if __name__ == "__main__":
    main()
