"""Recursos efectivos del Codespace para hilos CPU y diagnóstico."""
import math
import os
from pathlib import Path


def effective_cpu_count() -> int:
    cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            cpus = min(cpus, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, cpus)


def memory_limit_bytes() -> int | None:
    try:
        value = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if value != "max":
            limit = int(value)
            return limit if limit > 0 else None
    except (OSError, ValueError):
        pass
    return None


def configure_cpu_threads(mode: str = "train") -> int:
    """Entrenamiento usa todos los hilos efectivos; chat prioriza latencia."""
    import torch

    effective = effective_cpu_count()
    default = effective if mode == "train" else min(2, effective)
    configured = os.environ.get("ORIGEN_CPU_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if configured is None:
        threads = default
    else:
        try:
            threads = int(configured)
        except ValueError as exc:
            raise ValueError("ORIGEN_CPU_THREADS / OMP_NUM_THREADS deben ser enteros") from exc
        if threads <= 0:
            raise ValueError("El número de hilos CPU debe ser positivo")
        threads = min(threads, effective)
    torch.set_num_threads(threads)
    return threads
