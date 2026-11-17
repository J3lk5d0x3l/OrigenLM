"""Preentrenamiento de pesos ORIGINALES de OrigenLM con corpus binario grande.

Reanuda el checkpoint propio de 300 pasos; NO carga modelos externos.
Corpus uint16/mmap preparado mediante scripts/preparar_corpus.py.
Cada paso modifica pesos reales; checkpoints atómicos y validación separada.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from origen.model import ModelConfig, OrigenLM
from origen.resources import configure_cpu_threads


def read_shards(folder: Path, context: int):
    meta_file = folder / "meta.json"
    if not meta_file.is_file():
        raise ValueError(f"Corpus no preparado: {meta_file}. Ejecuta preparar_corpus primero.")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("format") != "origen-byte-v1":
        raise ValueError("Formato de tokenizador incompatible con el checkpoint original.")
    samples = {}
    for split in ("train", "val"):
        path = folder / (split + ".bin")
        if not path.is_file() or path.stat().st_size % 2:
            raise ValueError(f"Shard inexistente o corrupto: {path}")
        data = np.memmap(path, dtype="<u2", mode="r")
        docs = [d for d in meta["docs"] if d["split"] == split and d["tokens"] > context]
        weights = [d["tokens"] - context for d in docs]
        if not docs:
            raise ValueError(f"No existen documentos {split} con > {context} tokens. Reduce --seq-len.")
        samples[split] = (data, docs, weights)
    return samples, meta


def draw_batch(shard, context: int, batch_size: int, rng: random.Random, device):
    data, docs, weights = shard
    selection = rng.choices(docs, weights=weights, k=batch_size)
    positions = []
    for d in selection:
        # Un contexto debe pertenecer íntegramente a un documento.
        offset = rng.randrange(d["tokens"] - context)
        start = d["offset"] + offset
        positions.append(start)
    arr = np.stack([np.asarray(data[start:start + context + 1], dtype=np.int64) for start in positions])
    tensor = torch.from_numpy(arr).to(device)
    return tensor[:, :-1], tensor[:, 1:]


@torch.no_grad()
def evaluate(model, val_shard, context: int, samples: int, rng: random.Random, device):
    """Usar no_grad, no inference_mode: segura antes de volver al entrenamiento."""
    previous = model.training
    model.eval()
    total = 0.0
    try:
        for _ in range(samples):
            x, y = draw_batch(val_shard, context, 1, rng, device)
            logits, _ = model(x)
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
    finally:
        model.train(previous)
    return total / samples


def main():
    p = argparse.ArgumentParser(description="Entrenamiento masivo real de los pesos propios OrigenLM")
    p.add_argument("--corpus", type=Path, default=Path("dataset/preparado"))
    p.add_argument("--inicial", type=Path, default=Path("checkpoints/origen_latest.pt"))
    p.add_argument("--salida", type=Path, default=Path("checkpoints/origen_masivo.pt"))
    p.add_argument("--steps", type=int, default=20000, help="Máximo de pasos adicionales")
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--max-minutes", type=float, default=180, help="Guardar y parar tras este tiempo real")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.seq_len, args.batch_size, args.grad_accum, args.save_every, args.eval_batches) < 1:
        p.error("Todos los parámetros de tamaño deben ser positivos")
    if args.lr <= 0 or args.max_minutes <= 0:
        p.error("lr y max-minutes deben ser positivos")
    if args.inicial.resolve() == args.salida.resolve():
        p.error("La salida no puede sobrescribir el checkpoint histórico.")
    if not args.inicial.is_file():
        p.error(f"Checkpoint original no encontrado: {args.inicial}")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    threads = configure_cpu_threads("train")
    device = torch.device("cpu")
    shards, meta = read_shards(args.corpus, args.seq_len)
    # El checkpoint masivo es el único destino de reanudación, nunca el histórico.
    continue_from = args.salida if args.salida.is_file() else args.inicial
    saved = torch.load(continue_from, map_location="cpu", weights_only=True)
    if "architecture" in saved and saved["architecture"] != "origen-v1":
        p.error("Se necesita un checkpoint original de OrigenLM v1.")
    config = ModelConfig(**saved["config"])
    config.max_seq_len = max(config.max_seq_len, args.seq_len)
    model = OrigenLM(config).to(device)
    model.load_state_dict(saved["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if "optimizer" in saved:
        opt.load_state_dict(saved["optimizer"])
    for group in opt.param_groups:
        group["lr"] = args.lr
    step0 = int(saved.get("step", 0))
    training_rng = random.Random(args.seed + step0)
    validation_rng = random.Random(9999)
    began = time.monotonic()
    print(f"OrigenLM propio | CPU threads={threads} | parámetros={model.count_parameters():,}", flush=True)
    print(f"Checkpoint base: {continue_from} paso={step0}", flush=True)
    print(f"Corpus: {meta['total_documents']} documentos | {meta['total_tokens']:,} tokens", flush=True)
    print(f"Entrenamiento: batch={args.batch_size} x acumulación={args.grad_accum}, contexto={args.seq_len}", flush=True)
    initial_val = evaluate(model, shards["val"], args.seq_len, args.eval_batches, validation_rng, device)
    print(f"Validación inicial NLL={initial_val:.4f}", flush=True)
    tokens_since = 0
    last_log = time.monotonic()

    def save(step: int, train_loss: float):
        val_loss = evaluate(model, shards["val"], args.seq_len, args.eval_batches, validation_rng, device)
        payload = {"architecture": "origen-v1", "config": asdict(config), "step": step,
                   "model": model.state_dict(), "optimizer": opt.state_dict(),
                   "train_loss": train_loss, "val_loss": val_loss,
                   "objective": "causal_corpus_v1",
                   "corpus_tokens": int(meta["total_tokens"])}
        temp = args.salida.with_suffix(".pt.tmp")
        torch.save(payload, temp)
        os.replace(temp, args.salida)
        print(f"CHECKPOINT {args.salida} paso={step} | NLL_validación={val_loss:.4f}", flush=True)

    last_loss = math.nan
    last_step = step0
    try:
        for step in range(step0 + 1, step0 + args.steps + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(args.grad_accum):
                x, y = draw_batch(shards["train"], args.seq_len, args.batch_size, training_rng, device)
                _, loss = model(x, y)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Pérdida no finita en paso {step}: {loss.item()}")
                (loss / args.grad_accum).backward()
                loss_value += float(loss.detach()) / args.grad_accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss, last_step = loss_value, step
            tokens_since += args.seq_len * args.batch_size * args.grad_accum
            if step % 25 == 0 or step == step0 + 1:
                now = time.monotonic()
                elapsed = max(0.001, now - last_log)
                print(f"paso {step} | train_NLL={last_loss:.4f} | {tokens_since / elapsed:.0f} tokens/s | "
                      f"tiempo={int((now - began) / 60)}m", flush=True)
                last_log = now
                tokens_since = 0
            if step % args.save_every == 0:
                save(step, last_loss)
            if (time.monotonic() - began) / 60 >= args.max_minutes:
                if step % args.save_every != 0:
                    save(step, last_loss)
                print("Pausa por límite de tiempo, reanudable sin perder pesos.", flush=True)
                break
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido. Guardando checkpoint.", flush=True)
        if last_step > step0:
            save(last_step, last_loss)
    if last_step > step0 and last_step % args.save_every != 0 and args.salida.is_file():
        # Ya guardó el checkpoint por límite/interrupción: no duplicar innecesariamente.
        checkpoint_step = torch.load(args.salida, map_location="cpu", weights_only=True)["step"]
        if checkpoint_step != last_step:
            save(last_step, last_loss)
    elif last_step > step0 and not args.salida.is_file():
        save(last_step, last_loss)
    print(f"Finalizado en paso {last_step}. Pesos reales en {args.salida}", flush=True)


if __name__ == "__main__":
    main()
