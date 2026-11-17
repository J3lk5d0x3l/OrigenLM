"""A transparent UTF-8 byte tokenizer with no pretrained vocabulary.

All text symbols are raw byte values 0..255. The six special tokens
are ours; training requires no external tokenizer or downloaded assets.
"""
from typing import Iterable

BOS = 256
EOS = 257
SYSTEM = 258
USER = 259
ASSISTANT = 260
END = 261
VOCAB_SIZE = 262

# Misma instrucción durante entrenamiento e inferencia.
DEFAULT_SYSTEM = "Eres Origen, un asistente claro y útil."


def encode(text: str) -> list[int]:
    """Encode Unicode as UTF-8 bytes, tolerating malformed terminal input.

    Some terminals decode invalid input bytes as surrogate code points.
    These are not valid Unicode text and cannot be encoded with strict
    UTF-8; replacing only those invalid code points keeps chats running.
    Normal Spanish accents and emoji are preserved unchanged.
    """
    return list(text.encode("utf-8", errors="replace"))


def decode(ids: Iterable[int]) -> str:
    return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def training_example(item: dict) -> list[int]:
    """Encode either one original conversational turn or one plain-text record."""
    if "text" in item:
        return [BOS, *encode(str(item["text"])), EOS]
    if not all(key in item for key in ("user", "assistant")):
        raise ValueError("Record needs 'text' or both 'user' and 'assistant'.")
    return [
        BOS, SYSTEM, *encode(str(item.get("system", DEFAULT_SYSTEM)), END,
        USER, *encode(str(item["user"])), END,
        ASSISTANT, *encode(str(item["assistant"])), EOS,
    ]


def conversation_prefix(system: str, history: list[tuple[str, str]], user: str) -> list[int]:
    ids = [BOS, SYSTEM, *encode(system), END]
    for question, answer in history:
        ids += [USER, *encode(question), END, ASSISTANT, *encode(answer), END]
    ids += [USER, *encode(user), END, ASSISTANT]
    return ids

def conversation_prefix_limited(
    system: str, history: list[tuple[str, str]], user: str, max_seq_len: int
) -> list[int]:
    """Conserva la pregunta completa y solo los turnos anteriores que quepan.

    A diferencia del recorte implícito del generador, no pierde en silencio
    el mensaje del usuario ni la instrucción de sistema. El prototipo tiene
    un contexto de solo 96 bytes en sus checkpoints iniciales.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    selected: list[tuple[str, str]] = []
    base = conversation_prefix(system, selected, user)
    if len(base) > max_seq_len:
        raise ValueError(
            f"Tu mensaje excede el contexto del modelo ({len(base)} tokens "
            f"frente al máximo {max_seq_len}). Escribe una pregunta más corta."
        )
    for turn in reversed(history):
        candidate = conversation_prefix(system, [turn, *selected], user)
        if len(candidate) > max_seq_len:
            break
        selected.insert(0, turn)
    return conversation_prefix(system, selected, user)
