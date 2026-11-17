"""OrigenLM-Mythos: decoder Transformer ORIGINAL (sin pesos externos).

Arquitectura propia configurable: RMSNorm, RoPE, GQA, SwiGLU, SDPA,
pesos enlazados y contexto deslizante. Un nombre no implica las capacidades
ni la arquitectura privada de Claude Mythos.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from .tokenizer import VOCAB_SIZE, EOS, END


@dataclass
class MythosConfig:
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


MYTHOS_PRESETS = {
    "micro": dict(dim=96, n_layers=2, n_heads=4, n_kv_heads=2,
                  hidden_dim=256, max_seq_len=256),
    "student": dict(dim=256, n_layers=6, n_heads=8, n_kv_heads=2,
                    hidden_dim=768, max_seq_len=512),
    "l4": dict(dim=384, n_layers=8, n_heads=8, n_kv_heads=2,
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


def rope(q: torch.Tensor, k: torch.Tensor):
    """Aplicar RoPE a las cabezas de atención con el mismo eje temporal."""
    head_dim = q.size(-1)
    length = q.size(-2)
    frequencies = 1.0 / (10000.0 ** (
        torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim
    ))
    angles = torch.outer(torch.arange(length, device=q.device, dtype=torch.float32), frequencies)
    cos = angles.cos()[None, None, :, :].to(q.dtype)
    sin = angles.sin()[None, None, :, :].to(q.dtype)

    def rotate(x):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q, k)
        repeat = self.n_heads // self.n_kv_heads
        if repeat != 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MythosBlock(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class OrigenMythos(nn.Module):
    """Modelo autoregresivo desde cero; compatible con el tokenizador Origen."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(MythosBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.tokens.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2 or not 0 < tokens.size(1) <= self.config.max_seq_len:
            raise ValueError("tokens debe tener forma (batch, seq) dentro del contexto configurado")
        x = self.tokens(tokens)
        for block in self.layers:
            x = block(x)
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
                 temperature: float = 0.8, top_k: int = 40) -> list[int]:
        if not prefix:
            raise ValueError("El prompt no puede estar vacío")
        if max_new_tokens <= 0 or temperature < 0 or top_k < 0:
            raise ValueError("max_new_tokens debe ser positivo; temperature y top_k no negativos")
        device = next(self.parameters()).device
        generated = list(prefix)
        self.eval()
        for _ in range(max_new_tokens):
            tokens = torch.tensor([generated[-self.config.max_seq_len:]],
                                  dtype=torch.long, device=device)
            logits, _ = self(tokens)
            next_logits = logits[0, -1].float()
            if temperature == 0:
                new_token = int(next_logits.argmax().item())
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.numel()))
                    next_logits = next_logits.masked_fill(next_logits < values[-1], -torch.inf)
                new_token = int(torch.multinomial(F.softmax(next_logits, dim=-1), 1).item())
            if new_token in (EOS, END):
                break
            generated.append(new_token)
        return generated[len(prefix):]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
