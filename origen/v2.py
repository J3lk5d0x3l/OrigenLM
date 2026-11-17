"""OrigenLM v2: decoder Transformer ORIGINAL (sin pesos externos).

Arquitectura propia configurable: RMSNorm, RoPE, GQA, SwiGLU, SDPA,
pesos enlazados y contexto deslizante. Un nombre no implica las capacidades
ni la arquitectura privada de un modelo propietario.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from .tokenizer import VOCAB_SIZE, EOS, END


@dataclass
class OrigenV2Config:
    vocab_size: int = VOCAB_SIZE
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 768
    max_seq_len: int = 512
    norm_eps: float = 1e-6

    def validate(self):
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"El tokenizador requiere vocab_size={VOCAB_SIZE}")
        values = (self.dim, self.n_layers, self.n_heads, self.n_kv_heads,
                  self.hidden_dim, self.max_seq_len)
        if min(values) <= 0 or self.norm_eps <= 0:
            raise ValueError("Dimensiones y norm_eps deben ser positivos")
        if self.dim % self.n_heads:
            raise ValueError("dim debe ser divisible por n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads debe ser múltiplo de n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("La dimensión por cabeza debe ser par para RoPE")


ORIGEN_V2_PRESETS = {
    "micro": dict(dim=96, n_layers=2, n_heads=4, n_kv_heads=2,
                  hidden_dim=256, max_seq_len=256),
    "student": dict(dim=256, n_layers=6, n_heads=8, n_kv_heads=2,
                    hidden_dim=768, max_seq_len=512),
    "large": dict(dim=384, n_layers=8, n_heads=8, n_kv_heads=2,
               hidden_dim=1152, max_seq_len=1024),
    "research": dict(dim=768, n_layers=12, n_heads=12, n_kv_heads=3,
                     hidden_dim=2048, max_seq_len=2048),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (x.size(-1),), self.weight, eps=self.eps)


_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def rope_tables(head_dim: int, length: int, device, dtype):
    """Devuelve tablas cos/sin de RoPE reutilizadas entre capas y pasos.

    Las frecuencias no son parámetros aprendidos, así que calcularlas una vez
    por (head_dim, dispositivo, dtype) evita repetir arange/outer/cos/sin en
    cada capa de cada forward. La tabla crece bajo demanda.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim debe ser par y positivo para RoPE")
    if length <= 0:
        raise ValueError("length debe ser positivo")
    key = (head_dim, device, dtype)
    cached = _ROPE_CACHE.get(key)
    if cached is None or cached[0].size(-2) < length:
        frequencies = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        ))
        angles = torch.outer(
            torch.arange(length, device=device, dtype=torch.float32), frequencies
        )
        cached = (angles.cos()[None, None, :, :].to(dtype),
                  angles.sin()[None, None, :, :].to(dtype))
        _ROPE_CACHE[key] = cached
    cos, sin = cached
    return cos[..., :length, :], sin[..., :length, :]


def rope(q: torch.Tensor, k: torch.Tensor, offset: int = 0):
    """Aplicar RoPE a las cabezas de atención con el mismo eje temporal.

    `offset` indica cuántas posiciones previas hay en el KV-cache, de modo que
    un token generado reciba el mismo ángulo que tendría sin cache.
    """
    if offset < 0:
        raise ValueError("offset no puede ser negativo")
    head_dim = q.size(-1)
    length = q.size(-2)
    cos, sin = rope_tables(head_dim, length + offset, q.device, q.dtype)
    cos = cos[..., offset:offset + length, :]
    sin = sin[..., offset:offset + length, :]

    def rotate(x):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class KVCache:
    """Claves y valores ya calculados de UNA capa durante la generación.

    Guarda los tensores después de RoPE y antes de repetir los grupos de GQA,
    así que ocupa n_kv_heads en lugar de n_heads. No se usa en entrenamiento.
    """

    __slots__ = ("keys", "values")

    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.size(-2)

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat((self.keys, keys), dim=-2)
            self.values = torch.cat((self.values, values), dim=-2)
        return self.keys, self.values

    def trim(self, max_length: int):
        """Descarta los tokens más antiguos que exceden la ventana."""
        if self.keys is not None and self.keys.size(-2) > max_length:
            self.keys = self.keys[..., -max_length:, :]
            self.values = self.values[..., -max_length:, :]


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: OrigenV2Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cache: "KVCache | None" = None):
        batch, length, _ = x.shape
        offset = cache.length if cache is not None else 0
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q, k, offset=offset)
        if cache is not None:
            k, v = cache.append(k, v)
        repeat = self.n_heads // self.n_kv_heads
        if repeat != 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if offset == 0:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        elif length == 1:
            # Un solo token nuevo puede atender a todo el prefijo almacenado.
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            # is_causal alinea mal las esquinas cuando las longitudes difieren.
            total = k.size(-2)
            positions = torch.arange(offset, offset + length, device=x.device)
            mask = positions[:, None] >= torch.arange(total, device=x.device)[None, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask[None, None, :, :], dropout_p=0.0
            )
        return self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: OrigenV2Config):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class OrigenV2Block(nn.Module):
    def __init__(self, cfg: OrigenV2Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cache: "KVCache | None" = None):
        x = x + self.attn(self.attn_norm(x), cache)
        return x + self.ffn(self.ffn_norm(x))


class OrigenV2(nn.Module):
    """Modelo autoregresivo desde cero; compatible con el tokenizador Origen."""

    def __init__(self, cfg: OrigenV2Config):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(OrigenV2Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()
        self.lm_head.weight = self.tokens.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        """Reduce la desviación de las salidas que se suman al residual.

        Cada capa aporta dos sumas al flujo residual (atención y FFN). Sin
        escalar, la varianza del residual crece con la profundidad y los
        presets de 8 o 12 capas arrancan peor condicionados.
        """
        std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=std)

    def new_cache(self) -> list["KVCache"]:
        """Crea un KV-cache vacío, uno por capa, para generación incremental."""
        return [KVCache() for _ in self.layers]

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
                caches: "list[KVCache] | None" = None):
        if tokens.ndim != 2 or tokens.size(1) <= 0:
            raise ValueError("tokens debe tener forma (batch, seq) no vacía")
        offset = caches[0].length if caches else 0
        if offset + tokens.size(1) > self.config.max_seq_len:
            raise ValueError("la secuencia excede el contexto configurado")
        if caches is not None:
            if len(caches) != len(self.layers):
                raise ValueError("se requiere un KVCache por capa")
            if targets is not None:
                raise ValueError("el KV-cache es solo para inferencia, no acepta targets")
        x = self.tokens(tokens)
        for index, block in enumerate(self.layers):
            x = block(x, caches[index] if caches is not None else None)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("targets y tokens deben tener la misma forma")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(self, prefix: list[int], max_new_tokens: int = 160,
                 temperature: float = 0.8, top_k: int = 40,
                 use_cache: bool = True) -> list[int]:
        """Genera tokens nuevos reutilizando las claves/valores ya calculados.

        Sin cache el prefijo se reprocesaba en cada token (coste cuadrático).
        Con cache solo se evalúa el token nuevo, lo que hace usable el chat en
        CPU. `use_cache=False` conserva la ruta anterior para comparar.
        """
        if not prefix:
            raise ValueError("El prompt no puede estar vacío")
        if max_new_tokens <= 0 or temperature < 0 or top_k < 0:
            raise ValueError("max_new_tokens debe ser positivo; temperature y top_k no negativos")
        device = next(self.parameters()).device
        window = self.config.max_seq_len
        generated = list(prefix)
        self.eval()

        def choose(next_logits: torch.Tensor) -> int:
            next_logits = next_logits.float()
            if temperature == 0:
                return int(next_logits.argmax().item())
            next_logits = next_logits / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.numel()))
                next_logits = next_logits.masked_fill(next_logits < values[-1], -torch.inf)
            return int(torch.multinomial(F.softmax(next_logits, dim=-1), 1).item())

        if not use_cache:
            for _ in range(max_new_tokens):
                tokens = torch.tensor([generated[-window:]], dtype=torch.long, device=device)
                logits, _ = self(tokens)
                new_token = choose(logits[0, -1])
                if new_token in (EOS, END):
                    break
                generated.append(new_token)
            return generated[len(prefix):]

        caches = self.new_cache()
        tokens = torch.tensor([generated[-window:]], dtype=torch.long, device=device)
        logits, _ = self(tokens, caches=caches)
        for _ in range(max_new_tokens):
            new_token = choose(logits[0, -1])
            if new_token in (EOS, END):
                break
            generated.append(new_token)
            if caches[0].length >= window:
                # Ventana deslizante: libera la posición más antigua del cache.
                for cache in caches:
                    cache.trim(window - 1)
            step = torch.tensor([[new_token]], dtype=torch.long, device=device)
            logits, _ = self(step, caches=caches)
        return generated[len(prefix):]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
