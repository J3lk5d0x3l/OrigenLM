"""Evalúa OrigenLM con datos reservados sin modificar los pesos."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from origen.model import ModelConfig, OrigenLM
from origen.tokenizer import training_example


def read_examples(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el dataset: {path}")
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("el registro debe ser un objeto JSON")
            tokens = training_example(obj)
            if len(tokens) < 2:
                raise ValueError("el registro no contiene suficientes tokens")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Línea {line_number} inválida en {path}: {exc}") from exc
        examples.append(obj)
    if not examples:
        raise ValueError(f"Dataset vacío: {path}")
    return examples


@torch.inference_mode()
def evaluate(model: OrigenLM, examples: list[dict]) -> tuple[float, int]:
    """Pérdida next-token promediada por ejemplo independiente.

    El contexto se reinicia para cada ejemplo y se divide en fragmentos de
    hasta max_seq_len. No se modifican pesos ni optimizador.
    """
    model.eval()
    device = next(model.parameters()).device
    context = model.config.max_seq_len
    total_nll = 0.0
    total_predictions = 0
    for item in examples:
        ids = training_example(item)
        for start in range(0, len(ids) - 1, context):
            inputs = ids[start:start + context]
            targets = ids[start + 1:start + context + 1]
            if not targets:
                continue
            x = torch.tensor([inputs], dtype=torch.long, device=device)
            y = torch.tensor(targets, dtype=torch.long, device=device)
            logits, _ = model(x)
            total_nll += F.cross_entropy(logits[0, :len(targets)], y, reduction="sum").item()
            total_predictions += len(targets)
    return total_nll / total_predictions, total_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación independiente de OrigenLM (sin entrenar)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_latest.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/evaluacion_v2.jsonl"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint no encontrado: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = OrigenLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    examples = read_examples(args.data)
    loss, count = evaluate(model, examples)
    print(f"Checkpoint: paso {checkpoint['step']} | dispositivo: {device}")
    print(f"Evaluación independiente: {len(examples)} ejemplos, {count} tokens predichos")
    print(f"Pérdida promedio por byte/token (NLL): {loss:.4f} | perplejidad: {math.exp(loss):.2f}")
    print("Nota: la pérdida por byte no mide por sí sola la calidad de una conversación.")


if __name__ == "__main__":
    main()
