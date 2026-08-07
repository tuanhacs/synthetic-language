"""Llama-style decoder-only transformer.

One architecture for the whole family (context.md §7): pre-norm RMSNorm, RoPE
attention, SwiGLU MLP, no biases anywhere, tied embedding / lm_head, causal mask
via ``F.scaled_dot_product_attention``. No GQA, no MoE — the architecture stays
clean so scaling results are attributable to (N, D, language difficulty) alone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of shape ``(seq_len, head_dim // 2)`` for rotary embeddings."""
    if head_dim % 2:
        raise ValueError("RoPE needs an even head_dim")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, H, T, D)`` with tables sliced to ``T``."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        return x + self.drop(self.mlp(self.mlp_norm(x)))


class Model(nn.Module):
    """Decoder-only transformer over the 5-token bit vocabulary."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        cos, sin = rope_tables(cfg.context_len, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual-output projections (GPT-2 / Llama practice).
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """``idx`` of shape ``(B, T)`` -> logits ``(B, T, vocab)``."""
        _, t = idx.shape
        if t > self.cfg.context_len:
            raise ValueError(f"sequence length {t} exceeds context_len {self.cfg.context_len}")
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm(x))

    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count (the embedding is tiny: vocab_size * d_model)."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
        return total


def build_model(cfg: ModelConfig, tokenizer=None) -> Model:
    """Instantiate a model, asserting the vocabulary matches synthdata's tokenizer."""
    if tokenizer is None:
        import _paths  # noqa: F401  (sys.path shim for synthdata)
        from synthdata.tokenizer import BitTokenizer

        tokenizer = BitTokenizer()
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size {cfg.vocab_size} != BitTokenizer vocab_size {tokenizer.vocab_size}"
        )
    return Model(cfg)
