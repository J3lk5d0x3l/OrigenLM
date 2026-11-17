"""Diagnóstico de recursos de CPU para OrigenLM en GitHub Codespaces.

No consume GPU, no instala paquetes y no modifica el proyecto.
"""
import os
import platform
from pathlib import Path

def read(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return "no disponible"

def main():
    print("=== OrigenLM: diagnóstico de recursos Codespaces ===")
    print("Sistema:", platform.platform())
    print("Python:", platform.python_version())
    print("CPU (os.cpu_count):", os.cpu_count())
    if hasattr(os, "sched_getaffinity"):
        print("CPU (afinidad):", len(os.sched_getaffinity(0)))
    print("Cuota CPU cgroup:", read("/sys/fs/cgroup/cpu.max"))
    print("Límite memoria cgroup:", read("/sys/fs/cgroup/memory.max"))
    print("Memoria disponible cgroup:", read("/sys/fs/cgroup/memory.current"))
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("PyTorch threads:", torch.get_num_threads())
        print("CUDA disponible:", torch.cuda.is_available())
    except ImportError as exc:
        print("PyTorch sin instalar:", exc)
    print("No modificar la máquina ni iniciar entrenamientos masivos sin revisar presupuesto.")

if __name__ == "__main__":
    main()
