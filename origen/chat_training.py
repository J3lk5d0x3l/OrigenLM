"""Datos para enseñar a OrigenLM a producir la respuesta del asistente.

Cada conversación es independiente. Se ignora la pérdida sobre instrucciones
y preguntas: solamente el texto del asistente y EOS reciben supervisión.
Este código no descarga ni utiliza modelos preentrenados.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from origen.tokenizer import ASSISTANT, EOS, training_example

IGNORE_INDEX = -100


def read_chat_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el dataset: {path}")
    records = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: JSON inválido") from exc
        if not isinstance(item, dict) or not isinstance(item.get("user"), str) or not isinstance(item.get("assistant"), str):
            raise ValueError(f"{path}:{line_number}: se requieren 'user' y 'assistant' como textos")
        if not item["user"].strip() or not item["assistant"].strip():
            raise ValueError(f"{path}:{line_number}: la pregunta y respuesta no pueden estar vacías")
        question = item["user"].strip().casefold()
        if question in seen:
            raise ValueError(f"{path}:{line_number}: pregunta duplicada: {item['user']!r}")
        seen.add(question)
        records.append(item)
    if not records:
        raise ValueError(f"Dataset vacío: {path}")
    return records


def tokenize_response(item: dict, max_seq_len: int) -> tuple[list[int], list[int]]:
    """Tokeniza un ejemplo y enmascara todos los objetivos anteriores a la respuesta.

    inputs[pos] predice targets[pos]. Como ASSISTANT pertenece al prompt, el
    primer objetivo supervisado es el primer byte de la respuesta.
    """
    tokens = training_example(item)
    if len(tokens) - 1 > max_seq_len:
        raise ValueError(
            f"Ejemplo demasiado largo ({len(tokens)-1} > {max_seq_len} tokens): "
            f"{item['user'][:55]!r}. Aumenta --context o revisa el dataset."
        )
    assistant_pos = tokens.index(ASSISTANT)
    inputs = tokens[:-1]
    targets = tokens[1:]
    targets[:assistant_pos] = [IGNORE_INDEX] * assistant_pos
    assert targets[-1] == EOS
    return inputs, targets


def prepare_records(data: list[dict], max_seq_len: int) -> list[tuple[list[int], list[int]]]:
    return [tokenize_response(item, max_seq_len) for item in data]


def batchify(examples: list[tuple[list[int], list[int]]], device: torch.device):
    if not examples:
        raise ValueError("Batch vacío")
    width = max(len(x) for x, _ in examples)
    sources = torch.zeros((len(examples), width), dtype=torch.long, device=device)
    targets = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long, device=device)
    for index, (source, target) in enumerate(examples):
        sources[index, :len(source)] = torch.tensor(source, dtype=torch.long, device=device)
        targets[index, :len(target)] = torch.tensor(target, dtype=torch.long, device=device)
    return sources, targets


@torch.inference_mode()
def response_nll(model, examples: list[tuple[list[int], list[int]]]) -> tuple[float, int]:
    """NLL por byte de respuesta + EOS, nunca por prompt; sin alterar pesos."""
    if not examples:
        raise ValueError("No hay ejemplos de evaluación")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    count = 0
    try:
        for example in examples:
            sources, targets = batchify([example], device)
            logits, _ = model(sources)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            ).item()
            count += int((targets != IGNORE_INDEX).sum().item())
    finally:
        model.train(was_training)
    return loss_sum / count, count
