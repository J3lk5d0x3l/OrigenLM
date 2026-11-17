"""OrigenLM decoder-only Transformer, initializing EVERY weight at random.

Building blocks are standard published ML techniques: causal attention,
rotary position embeddings, RMSNorm and SwiGLU. None are pretrained models.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .tokenizer import VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    dim: int = 96
    n_layers: int = 3
    n_heads: int = 4
    hidden_dim: int = 256
    max_seq_len: int = 128

    def validate(self):
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if min(self.dim, self.n_layers, self.n_heads, self.hidden_dim, self.max_seq_len) <= 0:
            raise ValueError("All configuration dimensions must be > 0")


PRESETS = {
    "tiny": dict(dim=96, n_layers=3, n_heads=4, hidden_dim=256),
    "small": dict(dim=192, n_layers=4, n_heads=4, hidden_dim=512),
    "medium": dict(dim=384, n_layers=6, n_heads=6, hidden_dim=1024),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.scale)


def rotary_cos_sin(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
    return freqs.cos().to(dtype)[None, None], freqs.sin().to(dtype)[None, None]


def apply_rotary(x, cos, sin):
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    y = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return y.flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.output = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).split(dim, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        cos, sin = rotary_cos_sin(length, self.head_dim, x.device, q.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.output(out.transpose(1, 2).contiguous().view(batch, length, dim))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CausalAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class OrigenLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.tokens.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("Token input must be (batch, sequence) within max_seq_len")
        x = self.tokens(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, prefix: list[int], max_new_tokens: int = 80, temperature: float = 0.8, top_k: int = 30):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not prefix:
            raise ValueError("prefix cannot be empty")
        result = list(prefix)
        device = next(self.parameters()).device
        for _ in range(max_new_tokens):
            context = torch.tensor([result[-self.config.max_seq_len:]], device=device)
            logits, _ = self(context)
            last = logits[0, -1].float() / temperature
            if top_k > 0:
                values, _ = torch.topk(last, min(top_k, last.numel()))
                last[last < values[-1]] = -torch.inf
            probs = F.softmax(last, dim=-1)
            new_token = int(torch.multinomial(probs, 1).item())
            result.append(new_token)
            if new_token in (257, 261):
                break
        return result[len(prefix):]

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())
