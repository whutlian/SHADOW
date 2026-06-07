from __future__ import annotations

import os

import torch


def current_cpu_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def current_gpu_ram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())
