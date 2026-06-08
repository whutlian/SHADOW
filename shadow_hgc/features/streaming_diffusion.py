from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass
class StreamingDiffusionResult:
    block_paths: dict[str, Path]
    block_shapes: dict[str, tuple[int, int]]
    block_dtypes: dict[str, str]
    block_names: list[str]
    stats: dict[str, Any]


class _MemmapProvider:
    def __init__(self, path: Path, shape: tuple[int, int], dtype: str) -> None:
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.dim = shape[1]

    def get(self, indices: Any) -> np.ndarray:
        idx = _indices_to_numpy(indices)
        arr = np.memmap(self.path, mode="r", dtype=np.dtype(self.dtype), shape=self.shape)
        return np.asarray(arr[idx], dtype=np.float32)


class _ProjectedProvider:
    def __init__(self, provider: Any, weight: torch.Tensor) -> None:
        self.provider = provider
        self.weight = weight.cpu()
        self.dim = int(weight.shape[1])

    def get(self, indices: Any) -> torch.Tensor:
        x = _provider_get(self.provider, indices).to(dtype=self.weight.dtype)
        return x @ self.weight


def _indices_to_numpy(indices: Any) -> np.ndarray:
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy()
    return np.asarray(indices)


def _provider_dim(x_provider: Any) -> int:
    if isinstance(x_provider, torch.Tensor):
        return int(x_provider.shape[1])
    if hasattr(x_provider, "shape"):
        return int(x_provider.shape[1])
    if hasattr(x_provider, "dim"):
        return int(x_provider.dim)
    raise ValueError("x_provider must be a tensor/array or expose a dim attribute")


def _provider_get(x_provider: Any, indices: Any) -> torch.Tensor:
    if isinstance(x_provider, torch.Tensor):
        idx = indices.to(device=x_provider.device) if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, dtype=torch.long)
        return x_provider[idx].detach().cpu().to(dtype=torch.float32)
    if isinstance(x_provider, np.ndarray):
        return torch.as_tensor(x_provider[_indices_to_numpy(indices)], dtype=torch.float32)
    if hasattr(x_provider, "get"):
        return torch.as_tensor(x_provider.get(_indices_to_numpy(indices)), dtype=torch.float32)
    raise TypeError("x_provider must be a tensor/array or expose get(indices)")


def _make_projection(in_dim: int, out_dim: int, *, seed: int = 42) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(in_dim, out_dim, generator=generator, dtype=torch.float32)
    return weight / max(1.0, float(in_dim) ** 0.5)


def _write_memmap(path: Path, tensor: torch.Tensor, *, dtype: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    arr = np.memmap(path, mode="w+", dtype=np.dtype(dtype), shape=tuple(tensor.shape))
    arr[:] = tensor.detach().cpu().numpy().astype(np.dtype(dtype), copy=False)
    arr.flush()


def _empty_output(num_nodes: int, dim: int) -> torch.Tensor:
    return torch.zeros(num_nodes, dim, dtype=torch.float32)


def _diffuse_edge_chunked(
    *,
    x_provider: Any,
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    num_nodes: int,
    dim: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    out = _empty_output(num_nodes, dim)
    num_edges = int(edge_index.shape[1])
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        src = edge_index[0, start:end].cpu()
        dst = edge_index[1, start:end].cpu()
        src_feat = _provider_get(x_provider, src)
        weights = alpha[start:end].cpu().to(dtype=src_feat.dtype).unsqueeze(1)
        out.index_add_(0, dst, src_feat * weights)
    return out


def compute_streaming_diffusion_blocks(
    *,
    x_provider: Any,
    edge_index: Any = None,
    csr: Any = None,
    num_nodes: int,
    steps: tuple[int, ...] = (1,),
    include_highpass: bool = False,
    out_dir: str | Path,
    dtype: str = "float16",
    block_dim: int | None = None,
    edge_chunk_size: int = 1_000_000,
    dst_chunk_size: int | None = None,
    device: str = "cpu",
    normalize: str = "destination_row",
    overwrite: bool = False,
) -> StreamingDiffusionResult:
    """Compute destination-row-normalized diffusion blocks as bounded memmaps.

    The implemented backend streams by edge chunks. It materializes at most an
    `edge_chunk_size x feature_dim` message block per chunk, never the full
    edge-message matrix.
    """

    if csr is not None:
        raise NotImplementedError("csr diffusion is not implemented yet")
    if edge_index is None:
        raise ValueError("edge_index is required")
    if normalize != "destination_row":
        raise ValueError("only destination_row normalization is supported")
    if device != "cpu":
        raise ValueError("streaming diffusion currently supports device='cpu' only")
    if edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    edge_tensor = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    if edge_tensor.ndim != 2 or edge_tensor.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    in_dim = _provider_dim(x_provider)
    provider: Any = x_provider
    dim = in_dim
    if block_dim is not None and block_dim < in_dim:
        provider = _ProjectedProvider(x_provider, _make_projection(in_dim, int(block_dim)))
        dim = int(block_dim)

    alpha = destination_row_normalize(edge_tensor, num_nodes).to(dtype=torch.float32, device="cpu")
    requested = sorted({int(step) for step in steps if int(step) > 0})
    max_step = max(requested) if requested else 0

    block_paths: dict[str, Path] = {}
    block_shapes: dict[str, tuple[int, int]] = {}
    block_dtypes: dict[str, str] = {}
    block_names: list[str] = []
    current_provider = provider
    disk_bytes = 0

    for step in range(1, max_step + 1):
        out = _diffuse_edge_chunked(
            x_provider=current_provider,
            edge_index=edge_tensor,
            alpha=alpha,
            num_nodes=num_nodes,
            dim=dim,
            edge_chunk_size=int(edge_chunk_size),
        )
        name = f"X{step}"
        path = out_path / f"{name}.{np.dtype(dtype).name}.mmap"
        _write_memmap(path, out, dtype=dtype, overwrite=overwrite)
        shape = (int(num_nodes), int(dim))
        if step in requested:
            block_paths[name] = path
            block_shapes[name] = shape
            block_dtypes[name] = np.dtype(dtype).name
            block_names.append(name)
        disk_bytes += int(np.prod(shape) * np.dtype(dtype).itemsize)
        current_provider = _MemmapProvider(path, shape, np.dtype(dtype).name)

    if include_highpass:
        if max_step < 1:
            x1 = _diffuse_edge_chunked(
                x_provider=provider,
                edge_index=edge_tensor,
                alpha=alpha,
                num_nodes=num_nodes,
                dim=dim,
                edge_chunk_size=int(edge_chunk_size),
            )
        else:
            x1 = torch.as_tensor(
                np.asarray(np.memmap(out_path / f"X1.{np.dtype(dtype).name}.mmap", mode="r", dtype=np.dtype(dtype), shape=(num_nodes, dim))).copy(),
                dtype=torch.float32,
            )
        x0 = _empty_output(num_nodes, dim)
        for start in range(0, num_nodes, int(edge_chunk_size)):
            end = min(start + int(edge_chunk_size), num_nodes)
            idx = torch.arange(start, end, dtype=torch.long)
            x0[start:end] = _provider_get(provider, idx)
        hp = x0 - x1
        name = "Xhp"
        path = out_path / f"{name}.{np.dtype(dtype).name}.mmap"
        _write_memmap(path, hp, dtype=dtype, overwrite=overwrite)
        shape = (int(num_nodes), int(dim))
        block_paths[name] = path
        block_shapes[name] = shape
        block_dtypes[name] = np.dtype(dtype).name
        block_names.append(name)
        disk_bytes += int(np.prod(shape) * np.dtype(dtype).itemsize)

    stats = {
        "diffusion_backend": "edge_chunk",
        "diffusion_steps": requested,
        "include_highpass": bool(include_highpass),
        "diffusion_block_dim": dim,
        "diffusion_storage": f"{np.dtype(dtype).name}_memmap",
        "diffusion_disk_bytes": disk_bytes,
        "full_edge_scans": max_step + (1 if include_highpass and max_step < 1 else 0),
        "edge_chunk_size": int(edge_chunk_size),
        "dst_chunk_size": dst_chunk_size,
        "normalize": normalize,
    }
    return StreamingDiffusionResult(block_paths, block_shapes, block_dtypes, block_names, stats)
