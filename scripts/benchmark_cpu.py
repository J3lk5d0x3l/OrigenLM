"""Comparar rendimiento real del entrenador de OrigenLM en CPU.

Benchmarks breves sobre tensores sintéticos; no entrena ni sobrescribe el
checkpoint real, no modifica corpus ni conecta proveedores externos.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch

from origen.model import ModelConfig, OrigenLM
from origen.resources import effective_cpu_count


def measure(model, seq_len: int, microbatch: int, accumulation: int,
            trials: int, device: torch.device) -> tuple[float, float]:
    """Mide forward, backward, acumulación, clipping y paso AdamW."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Datos sintéticos fijos: medir cálculo, no la velocidad del disco.
    torch.manual_seed(123)
    x = torch.randint(0, 256, (microbatch, seq_len), device=device)
    y = torch.randint(0, 256, (microbatch, seq_len), device=device)

    def update() -> None:
        opt.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            _, loss = model(x, y)
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    update()  # warmup para evitar incluir inicialización en la medida
    start = time.perf_counter()
    for _ in range(trials):
        update()
    seconds = time.perf_counter() - start
    throughput = microbatch * accumulation * seq_len * trials / max(seconds, 1e-9)
    return throughput, seconds / trials


def main(argv=None):
    p = argparse.ArgumentParser(description="Benchmark CPU del modelo propio OrigenLM")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_masivo.pt"))
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args(argv)
    if min(args.seq_len, args.effective_batch, args.trials) < 1:
        p.error("seq-len, effective-batch y trials deben ser positivos")
    if not args.checkpoint.is_file():
        other = Path("checkpoints/origen_latest.pt")
        if not other.is_file():
            p.error("No hay checkpoint original; verifica origen estado")
        args.checkpoint = other
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("architecture") not in (None, "origen-v1"):
        p.error("Este benchmark utiliza exclusivamente pesos propios v1.")
    original_cfg = ModelConfig(**saved["config"])
    cfg = replace(original_cfg, max_seq_len=max(original_cfg.max_seq_len, args.seq_len))
    usable = effective_cpu_count()
    cpu_options = sorted(set((1, max(1, usable // 2), usable)))
    batch_options = [b for b in (4, 8, 16) if b <= args.effective_batch
                     and args.effective_batch % b == 0]
    if not batch_options:
        batch_options = [args.effective_batch]
    print(f"OrigenLM CPU benchmark | núcleos efectivos={usable} | contexto={args.seq_len}", flush=True)
    print(f"Checkpoint leído solo para benchmark: {args.checkpoint}", flush=True)
    print("Los tokens/s del benchmark sintético NO garantizan igual velocidad con corpus real.", flush=True)
    rows = []
    for threads in cpu_options:
        torch.set_num_threads(threads)
        for batch_size in batch_options:
            accum = args.effective_batch // batch_size
            model = OrigenLM(cfg)
            model.load_state_dict(saved["model"])
            speed, per_step = measure(model, args.seq_len, batch_size, accum,
                                      args.trials, torch.device("cpu"))
            rows.append((speed, threads, batch_size, accum, per_step))
            print(f"hilos={threads:2} batch={batch_size:2} acum={accum:2} | "
                  f"{speed:,.0f} tokens/s | {per_step:.3f} s/paso", flush=True)
            del model
    best = max(rows)
    print(f"Mejor configuración medida: hilos={best[1]} batch={best[2]} "
          f"acumulación={best[3]} | {best[0]:,.0f} tokens/s", flush=True)
    print("Para usarla en la siguiente sesión ajusta ORIGEN_CPU_THREADS, "
          "--batch-size y --grad-accum sin cambiar el batch efectivo.")


if __name__ == "__main__":
    main()
