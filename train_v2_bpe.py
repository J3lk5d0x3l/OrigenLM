"""Preentrenamiento V2 BPE desde pesos aleatorios sobre bins uint16.

Lee `dataset/preparado_v2_es (o cualquier carpeta con formato origen-bpe-v2)
con mmap, muestrea ventanas aleatorias y evalúa la validación en un tramo fijo
para que la métrica sea comparable entre pasos.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from origen.resources import configure_cpu_threads
from origen.v2 import OrigenV2, OrigenV2Config


def cargar_bins(carpeta: Path):
    meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != "origen-bpe-v2":
        raise ValueError(f"{carpeta}/meta.json no tiene formato origen-bpe-v2")
    datos = {s: np.memmap(carpeta / d["path"], dtype="<u2", mode="r")
             for s, d in meta["splits"].items()}
    return meta, datos


def muestrear(flujo, ctx: int, batch: int, gen: np.random.Generator):
    """Ventanas aleatorias sin reutilizar el mismo inicio dos veces por lote."""
    inicios = gen.integers(0, len(flujo) - ctx - 2, size=batch)
    entradas = np.empty((batch, ctx), dtype=np.int64)
    objetivos = np.empty((batch, ctx), dtype=np.int64)
    for fila, inicio in enumerate(inicios):
        entradas[fila] = flujo[inicio:inicio + ctx]
        objetivos[fila] = flujo[inicio + 1:inicio + 1 + ctx]
    return torch.from_numpy(entradas), torch.from_numpy(objetivos)


@torch.no_grad()
def evaluar(modelo, flujo, ctx: int, batch: int, max_tokens: int = 200_000):
    """NLL medio por token sobre un tramo contiguo y fijo de validación.

    Se usa no_grad (no inference_mode): las tablas de RoPE se guardan en caché
    y unos tensores de inferencia reutilizados en entrenamiento rompen autograd.
    """
    estaba_entrenando = modelo.training
    modelo.eval()
    total = 0.0
    vistos = 0
    inicio = 0
    try:
        while vistos < max_tokens and inicio + batch * ctx + 1 <= len(flujo):
            x = np.empty((batch, ctx), dtype=np.int64)
            y = np.empty((batch, ctx), dtype=np.int64)
            for fila in range(batch):
                desde = inicio + fila * ctx
                x[fila] = flujo[desde:desde + ctx]
                y[fila] = flujo[desde + 1:desde + 1 + ctx]
            _, perdida = modelo(torch.from_numpy(x), torch.from_numpy(y))
            total += float(perdida) * x.size
            vistos += x.size
            inicio += batch * ctx
    finally:
        modelo.train(estaba_entrenando)
    if not vistos:
        raise ValueError("Validación demasiado corta para el contexto pedido")
    return total / vistos, vistos


def main() -> int:
    p = argparse.ArgumentParser(description="Preentrenar OrigenLM V2 (BPE) en CPU")
    p.add_argument("--datos", type=Path, default=Path("dataset/preparado_v2_es"))
    p.add_argument("--config", type=Path, default=Path("configs/origen_es_7m.json"))
    p.add_argument("--salida", type=Path, default=Path("checkpoints/origen_v2_es.pt"))
    p.add_argument("--pasos", type=int, default=20000)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--guardar-cada", type=int, default=500)
    p.add_argument("--eval-cada", type=int, default=250)
    p.add_argument("--max-minutos", type=float, default=0.0, help="0 = sin límite")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if min(args.pasos, args.ctx, args.batch_size, args.grad_accum,
           args.guardar_cada, args.eval_cada) < 1 or args.lr <= 0:
        p.error("pasos, ctx, batch, acumulación, guardado, eval y lr deben ser positivos")
    if args.salida.exists() and not (args.overwrite or args.resume):
        p.error("El checkpoint ya existe: usa --resume o --overwrite")

    hilos = configure_cpu_threads("train")
    meta, datos = cargar_bins(args.datos)
    train, val = datos["train"], datos["val"]
    if args.resume:
        guardado = torch.load(args.resume, map_location="cpu", weights_only=True)
        if guardado.get("architecture") != "origen-v2-gqa-v1":
            p.error("El checkpoint no es de OrigenLM v2")
        cfg = OrigenV2Config(**guardado["config"])
        inicio_paso = int(guardado["step"])
    else:
        cfg = OrigenV2Config(**json.loads(args.config.read_text(encoding="utf-8")))
        guardado = None
        inicio_paso = 0
    if args.ctx > cfg.max_seq_len:
        p.error(f"--ctx {args.ctx} supera max_seq_len {cfg.max_seq_len}")
    cfg.validate()

    torch.manual_seed(args.seed)
    modelo = OrigenV2(cfg)
    if guardado:
        modelo.load_state_dict(guardado["model"])
    decay, sin_decay = [], []
    for nombre, parametro in modelo.named_parameters():
        (sin_decay if parametro.ndim < 2 or "tokens" in nombre else decay).append(parametro)
    optimizador = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": sin_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=False)
    if guardado and "optimizer" in guardado:
        optimizador.load_state_dict(guardado["optimizer"])

    batch_efectivo = args.batch_size * args.grad_accum
    tokens_paso = batch_efectivo * args.ctx
    tokens_objetivo = args.pasos * tokens_paso
    print(f"OrigenLM V2 ES | {modelo.count_parameters():,} parámetros | hilos={hilos}")
    print(f"datos: {meta['total_tokens']:,} tokens | vocab={meta['vocab_size']} | "
          f"ctx={args.ctx} | batch efectivo={batch_efectivo} ({tokens_paso:,} tokens/paso)")
    print(f"objetivo: {args.pasos:,} pasos = {tokens_objetivo / 1e6:.1f}M tokens "
          f"({tokens_objetivo / max(1, len(train)):.1f} épocas)")
    if args.resume:
        print(f"continuando desde paso {inicio_paso}")
    inicio = time.time()
    nll_val, tokens_val = evaluar(modelo, val, args.ctx, args.batch_size)
    print(f"validación inicial: NLL={nll_val:.4f} ({tokens_val:,} tokens)", flush=True)

    gen = np.random.default_rng(args.seed + inicio_paso)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    tokens_vistos = 0
    marca = time.time()
    for paso in range(inicio_paso + 1, inicio_paso + args.pasos + 1):
        hecho = paso - inicio_paso
        if hecho <= args.warmup:
            lr = args.lr * hecho / max(1, args.warmup)
        else:
            avance = (hecho - args.warmup) / max(1, args.pasos - args.warmup)
            lr = args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio)
                            * 0.5 * (1 + math.cos(math.pi * min(1.0, avance))))
        for grupo in optimizador.param_groups:
            grupo["lr"] = lr
        modelo.train()
        optimizador.zero_grad(set_to_none=True)
        perdida_total = 0.0
        for _ in range(args.grad_accum):
            x, y = muestrear(train, args.ctx, args.batch_size, gen)
            _, perdida = modelo(x, y)
            (perdida / args.grad_accum).backward()
            perdida_total += float(perdida.detach()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        optimizador.step()
        tokens_vistos += tokens_paso

        if paso % 20 == 0 or paso == inicio_paso + 1:
            segundos = max(1e-6, time.time() - marca)
            ritmo = tokens_paso * 20 / segundos if paso % 20 == 0 else 0.0
            marca = time.time()
            restante = (inicio_paso + args.pasos - paso) * tokens_paso
            eta = restante / ritmo / 60 if ritmo else 0.0
            print(f"paso {paso:6d} | NLL={perdida_total:.4f} | lr={lr:.2e} | "
                  f"{ritmo:,.0f} tok/s | ETA {eta:.0f} min", flush=True)

        if paso % args.eval_cada == 0 or paso == inicio_paso + args.pasos:
            nll_val, tokens_val = evaluar(modelo, val, args.ctx, args.batch_size)
            print(f"  validación: NLL={nll_val:.4f} ({tokens_val:,} tokens) | "
                  f"perplejidad={math.exp(min(20, nll_val)):.2f}", flush=True)

        if paso % args.guardar_cada == 0 or paso == inicio_paso + args.pasos:
            carga = {
                "architecture": "origen-v2-gqa-v1",
                "config": cfg.__dict__.copy(),
                "step": paso,
                "model": modelo.state_dict(),
                "optimizer": optimizador.state_dict(),
                "train_loss": perdida_total,
                "val_loss": nll_val,
                "tokens_vistos": inicio_paso * tokens_paso + tokens_vistos,
                "data": str(args.datos),
                "tokenizer_sha256": meta["tokenizer_sha256"],
            }
            temporal = args.salida.with_name(args.salida.name + ".tmp")
            torch.save(carga, temporal)
            os.replace(temporal, args.salida)
            print(f"  guardado: {args.salida} | paso {paso}", flush=True)

        if args.max_minutos and (time.time() - inicio) / 60 >= args.max_minutos:
            print(f"Límite de {args.max_minutos:.0f} min alcanzado en el paso {paso}.")
            break

    print("Terminado. Compara respuestas reales antes de asumir mejora.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
