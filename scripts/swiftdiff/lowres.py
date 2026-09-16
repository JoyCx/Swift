"""Shared resource limits for heavy local jobs on the desktop machine (Windows, single RTX 5090)."""
import ctypes
import os

import torch


def limit(gpu_frac: float = 0.6, threads: int = 6) -> None:
    """Background CPU/IO/memory priority, fixed CPU threads, hard per-process GPU memory cap.

    PROCESS_MODE_BACKGROUND_BEGIN lowers CPU, disk I/O and memory-page priority together, so the desktop
    keeps precedence. The CUDA cap turns over-allocation into an OOM error instead of Windows spilling GPU
    allocations into shared system memory (which stalls every GPU app).
    """
    if os.name == "nt":
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.kernel32.SetPriorityClass(h, 0x00100000):  # PROCESS_MODE_BACKGROUND_BEGIN
            ctypes.windll.kernel32.SetPriorityClass(h, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    torch.set_num_threads(threads)
    if torch.cuda.is_available() and torch.cuda.device_count():
        torch.cuda.set_per_process_memory_fraction(gpu_frac, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[lowres] GPU cap {gpu_frac:.0%} = {gpu_frac * total:.1f} GiB of {total:.1f} GiB, {threads} CPU threads, background priority", flush=True)
