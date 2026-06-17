"""Compatibility launcher for cloned scalable external baselines.

The external repositories are kept as-is. This wrapper only patches runtime API
aliases needed by the local conda environment before delegating to their entry
points.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import runpy
import sys
from collections.abc import Sequence
from typing import Any


def patch_dgl_message_api(fn_module: Any) -> None:
    """Add old DGL message-function aliases expected by some external repos."""

    if not hasattr(fn_module, "copy_src") and hasattr(fn_module, "copy_u"):
        def copy_src(src: str, out: str):
            return fn_module.copy_u(src, out)

        fn_module.copy_src = copy_src
    if not hasattr(fn_module, "copy_edge") and hasattr(fn_module, "copy_e"):
        def copy_edge(edge: str, out: str):
            return fn_module.copy_e(edge, out)

        fn_module.copy_edge = copy_edge


def patch_numpy_bool(np_module: Any) -> None:
    """Restore the removed ``np.bool`` alias for old GraphSAINT code."""

    if not hasattr(np_module, "bool") and hasattr(np_module, "bool_"):
        np_module.bool = np_module.bool_


def build_entry_argv(entry: str, entry_args: Sequence[str]) -> list[str]:
    return [entry, *entry_args]


def ensure_entry_parent_on_path(entry: str) -> None:
    parent = str(Path(entry).parent)
    if parent and parent != "." and parent not in sys.path:
        sys.path.insert(0, parent)


def ensure_cwd_on_path() -> None:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def make_safe_cuda_device(original_device):
    def safe_cuda_device(device=None):
        if str(device).startswith("cpu"):
            return contextlib.nullcontext()
        return original_device(device)

    return safe_cuda_device


def run_sagn(entry: str, entry_args: Sequence[str]) -> None:
    import dgl.function as fn
    import torch

    patch_dgl_message_api(fn)
    torch.cuda.device = make_safe_cuda_device(torch.cuda.device)
    ensure_entry_parent_on_path(entry)
    sys.argv = build_entry_argv(entry, entry_args)
    runpy.run_path(entry, run_name="__main__")


def run_graphsaint(entry_args: Sequence[str]) -> None:
    import numpy as np

    patch_numpy_bool(np)
    ensure_cwd_on_path()
    sys.argv = build_entry_argv("graphsaint.pytorch_version.train", entry_args)
    runpy.run_module("graphsaint.pytorch_version.train", run_name="__main__")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalable external baselines with environment compatibility patches.")
    parser.add_argument("--mode", choices=("sagn", "graphsaint"), required=True)
    parser.add_argument("--entry", default=None, help="Script path for --mode sagn. Ignored by graphsaint.")
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.entry_args and args.entry_args[0] == "--":
        args.entry_args = args.entry_args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "sagn":
        run_sagn(args.entry or "src/sagn.py", args.entry_args)
    elif args.mode == "graphsaint":
        run_graphsaint(args.entry_args)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
