"""Reentrenamiento conversacional de OrigenLM a partir del checkpoint propio.

Crea un checkpoint diferente: checkpoints/origen_respuestas.pt. Nunca modifica
origen_latest.pt ni los respaldos de 200/300 pasos.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch

from origen.chat_training import batchify, prepare_records, read_chat_dataset, response_nll
from origen.model import ModelConfig, OrigenLM, PRESETS
from origen.resources import configure_cpu_threads


def main():
    parser = argparse.ArgumentParser(description="Entrenar respuestas de OrigenLM desde los pesos propios")
    parser.add_argument("--resume", type=Path, default=Path("checkpoints/origen_latest.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/origen_respuestas.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/chat_v2.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("dataset/evaluacion_v2.jsonl"))
    parser.add_argument("--steps", type=int, default=100, help="Pasos adicionales; no pasos totales")
    parser.add_argument("--context", type=int, default=384, help="Máximo de bytes/tokens por ejemplo")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=PRESETS, default="tiny", help="Solo se usa sin --resume")
    args = parser.parse_args()

    if min(args.steps, args.context, args.batch_size, args.save_every) <= 0 or args.lr <= 0:
        parser.error("steps, context, batch-size, save-every y lr deben ser positivos")
    if args.resume.resolve() == args.output.resolve():
        parser.error("--output no puede sobrescribir el checkpoint original de --resume")
    if not args.resume.is_file():
        parser.error(f"No existe el checkpoint: {args.resume}")

    torch.manual_seed(args.seed)
    configure_cpu_threads("train")
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**saved["config"])
    # RoPE se calcula dinámicamente; ampliar la ventana no cambia la forma
    # de los pesos aprendidos. No implica que el modelo ya sepa usarla.
    cfg.max_seq_len = max(cfg.max_seq_len, args.context)
    model = OrigenLM(cfg)
    model.load_state_dict(saved["model"])
    model.to(device)

    training = read_chat_dataset(args.data)
    validation = read_chat_dataset(args.validation)
    train_questions = {item["user"].strip().casefold() for item in training}
    val_questions = {item["user"].strip().casefold() for item in validation}
    overlap = train_questions & val_questions
    if overlap:
        parser.error(f"Preguntas repetidas en entrenamiento y validación: {sorted(overlap)[:3]}")
    train_samples = prepare_records(training, cfg.max_seq_len)
    val_samples = prepare_records(validation, cfg.max_seq_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if saved.get("objective") == "assistant_only_v1" and "optimizer" in saved:
        optimizer.load_state_dict(saved["optimizer"])
    start = int(saved.get("step", 0))
    initial_loss, initial_count = response_nll(model, val_samples)
    print(f"OrigenLM | dispositivo={device} | parámetros={model.count_parameters():,}")
    print(f"Entrenamiento: {len(training)} conversaciones | validación: {len(validation)} conversaciones")
    print(f"Contexto: {cfg.max_seq_len} tokens | objetivo: respuesta + EOS")
    print(f"Validación ANTES del entrenamiento: NLL={initial_loss:.4f} ({initial_count} tokens)")
    print("Advertencia: 133 ejemplos son una prueba técnica, no un corpus para un asistente general.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(start + 1, start + args.steps + 1):
        chosen = rng.choices(train_samples, k=args.batch_size)
        sources, targets = batchify(chosen, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(sources, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == start + 1 or step % 10 == 0:
            print(f"paso {step:5d} | pérdida de respuestas {loss.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == start + args.steps:
            val_loss, val_count = response_nll(model, val_samples)
            payload = {
                "config": asdict(cfg), "step": step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "objective": "assistant_only_v1",
                "loss": float(loss.item()), "val_loss": val_loss,
            }
            torch.save(payload, args.output)
            print(f"Guardado: {args.output} (paso {step}) | NLL validación={val_loss:.4f} ({val_count} tokens)")
    print("La evaluación solo refleja estos 15 ejemplos reservados; revisa también las respuestas generadas.")


if __name__ == "__main__":
    main()
