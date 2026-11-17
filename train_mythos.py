"""Entrenar OrigenLM-Mythos desde pesos aleatorios; no usa modelos externos."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import torch

from origen.chat_training import batchify, prepare_records, read_chat_dataset, response_nll
from origen.mythos import MYTHOS_PRESETS, MythosConfig, OrigenMythos


def main():
    parser = argparse.ArgumentParser(description="Entrenar OrigenLM-Mythos sin pesos preentrenados")
    parser.add_argument("--preset", choices=MYTHOS_PRESETS, default="micro")
    parser.add_argument("--config", type=Path, help="JSON con parámetros propios, en vez del preset")
    parser.add_argument("--data", type=Path, default=Path("dataset/chat_v2.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("dataset/evaluacion_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mythos_latest.pt"))
    parser.add_argument("--resume", type=Path, help="Checkpoint Mythos anterior")
    parser.add_argument("--steps", type=int, default=10, help="Pasos adicionales")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Permitir reemplazar un destino existente")
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.grad_accum, args.save_every) < 1 or args.lr <= 0:
        parser.error("steps, batch-size, grad-accum, save-every y lr deben ser positivos")
    if args.resume and args.config:
        parser.error("--config no se usa junto con --resume: la arquitectura viene del checkpoint")
    if args.output.exists() and not (args.overwrite or (args.resume and args.resume.resolve() == args.output.resolve())):
        parser.error("El checkpoint de salida ya existe. Usa --resume con esa ruta o --overwrite.")
    if args.resume and not args.resume.is_file():
        parser.error(f"Checkpoint no encontrado: {args.resume}")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("architecture") != "origen-mythos-v1":
            parser.error("El checkpoint no pertenece a OrigenLM-Mythos v1")
        cfg = MythosConfig(**saved["config"])
    elif args.config:
        cfg = MythosConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        saved = None
    else:
        cfg = MythosConfig(**MYTHOS_PRESETS[args.preset])
        saved = None
    cfg.validate()

    training = read_chat_dataset(args.data)
    validation = read_chat_dataset(args.validation)
    questions = {item["user"].strip().casefold() for item in training}
    overlap = questions & {item["user"].strip().casefold() for item in validation}
    if overlap:
        parser.error(f"Hay preguntas duplicadas entre entrenamiento y validación: {sorted(overlap)[:3]}")
    training_samples = prepare_records(training, cfg.max_seq_len)
    validation_samples = prepare_records(validation, cfg.max_seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrigenMythos(cfg).to(device)
    if saved:
        model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if saved and "optimizer" in saved:
        optimizer.load_state_dict(saved["optimizer"])
    start_step = int(saved["step"]) if saved else 0
    rng = random.Random(args.seed + start_step)
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

    print(f"OrigenLM-Mythos | dispositivo={device} | parámetros={model.count_parameters():,}")
    print(f"Ventana de contexto: {cfg.max_seq_len} tokens | ejemplos: {len(training)} | validación: {len(validation)}")
    print("Pesos originales aleatorios." if not saved else f"Continuando checkpoint propio desde paso {start_step}.")
    print("Advertencia: los datos de demostración no producen una IA de capacidades avanzadas.")
    initial_nll, n = response_nll(model, validation_samples)
    print(f"NLL de respuestas antes de entrenar: {initial_nll:.4f} ({n} tokens)")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            selected = rng.choices(training_samples, k=args.batch_size)
            inputs, labels = batchify(selected, device)
            with precision:
                _, loss = model(inputs, labels)
            (loss / args.grad_accum).backward()
            total_loss += float(loss.detach()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == start_step + 1 or step % 10 == 0:
            print(f"paso {step:5d} | NLL batch de respuestas: {total_loss:.4f}", flush=True)
        if step % args.save_every == 0 or step == start_step + args.steps:
            val_loss, val_count = response_nll(model, validation_samples)
            payload = {
                "architecture": "origen-mythos-v1", "config": asdict(cfg),
                "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": total_loss, "val_loss": float(val_loss),
            }
            temporary = args.output.with_name(args.output.name + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, args.output)
            print(f"Guardado: {args.output} | paso {step} | NLL validación: {val_loss:.4f} ({val_count} tokens)")
    print("Finalizado. Revisa pruebas de conversación antes de concluir que el modelo mejora.")


if __name__ == "__main__":
    main()
