"""Comparar el error sobre respuestas para checkpoints viejos y nuevos.

Es comparable dentro del mismo conjunto y con la misma ventana de contexto.
No entrena ni modifica los checkpoints.
"""
import argparse
from pathlib import Path

import torch

from origen.chat_training import prepare_records, read_chat_dataset, response_nll
from origen.model import ModelConfig, OrigenLM


def main():
    parser = argparse.ArgumentParser(description="NLL de respuestas de OrigenLM")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_latest.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/evaluacion_v2.jsonl"))
    parser.add_argument("--context", type=int, default=384)
    args = parser.parse_args()
    if args.context <= 0:
        parser.error("--context debe ser positivo")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint no encontrado: {args.checkpoint}")

    stored = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**stored["config"])
    cfg.max_seq_len = max(cfg.max_seq_len, args.context)
    model = OrigenLM(cfg)
    model.load_state_dict(stored["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    samples = prepare_records(read_chat_dataset(args.data), cfg.max_seq_len)
    nll, n = response_nll(model, samples)
    print(f"Checkpoint: paso {stored['step']} | dispositivo: {device}")
    print(f"Evaluación de RESPUESTAS: {len(samples)} ejemplos, {n} tokens")
    print(f"NLL por token de respuesta: {nll:.4f}")
    print("Mide predicción de respuestas, no comprensión ni exactitud de los saludos.")


if __name__ == "__main__":
    main()
