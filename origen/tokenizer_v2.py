"""Tokenizador BPE propio de OrigenLM V2, sin vocabularios ni pesos externos."""
from pathlib import Path

SPECIAL = ("<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
BOS, EOS, SYSTEM, USER, ASSISTANT, END = range(6)


def library():
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError as exc:
        raise RuntimeError("Instala tokenizers>=0.20 para preparar V2") from exc
    return Tokenizer, models, pre_tokenizers, decoders, trainers


def create_tokenizer():
    Tokenizer, models, pre_tokenizers, decoders, _ = library()
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def validate_tokenizer(tokenizer):
    for token_id, spelling in enumerate(SPECIAL):
        if tokenizer.token_to_id(spelling) != token_id:
            raise ValueError(f"Token especial incompatible: {spelling}")
    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("El vocabulario no cabe en uint16")
    for text in ("¡Hola, Colombia!", "def sumar(a, b): return a + b", "áéíóú ñ Ñ 😀"):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError("El tokenizador no puede reconstruir UTF-8")


def train_tokenizer(texts, vocab_size=8192, min_frequency=2):
    if not 512 <= vocab_size <= 65535 or min_frequency < 1:
        raise ValueError("vocab 512..65535 y min_frequency positivo")
    _, _, pre_tokenizers, _, trainers = library()
    tok = create_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(SPECIAL), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    validate_tokenizer(tok)
    return tok


def load_tokenizer(path: Path):
    Tokenizer, *_ = library()
    tok = Tokenizer.from_file(str(path))
    validate_tokenizer(tok)
    return tok


def encode_document(tok, text):
    return [BOS, *tok.encode(text, add_special_tokens=False).ids, EOS]
