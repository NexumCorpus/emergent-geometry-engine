"""Minimal 1-layer transformer for modular addition (Nanda-style grokking recipe).

Hand-rolled (no TransformerLens), NO LayerNorm, attention + MLP, reads the last ('=')
position. W_E is exposed directly — it is the carrier of the Fourier-circle structure
the geometric order parameter measures.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerTransformer(nn.Module):
    def __init__(self, d_vocab: int, d_vocab_out: int, d_model: int = 128, n_ctx: int = 3,
                 n_heads: int = 4, d_head: int = 32, d_mlp: int = 512, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def randn(*shape, scale):
            return nn.Parameter(torch.randn(*shape, generator=g) * scale)

        self.n_heads, self.d_head, self.n_ctx = n_heads, d_head, n_ctx
        self.W_E = randn(d_vocab, d_model, scale=1.0 / math.sqrt(d_model))
        self.W_pos = randn(n_ctx, d_model, scale=1.0 / math.sqrt(d_model))
        self.W_Q = randn(n_heads, d_model, d_head, scale=1.0 / math.sqrt(d_model))
        self.W_K = randn(n_heads, d_model, d_head, scale=1.0 / math.sqrt(d_model))
        self.W_V = randn(n_heads, d_model, d_head, scale=1.0 / math.sqrt(d_model))
        self.W_O = randn(n_heads, d_head, d_model, scale=1.0 / math.sqrt(d_model))
        self.W_in = randn(d_model, d_mlp, scale=1.0 / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = randn(d_mlp, d_model, scale=1.0 / math.sqrt(d_mlp))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.W_U = randn(d_model, d_vocab_out, scale=1.0 / math.sqrt(d_model))
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.W_E[tokens] + self.W_pos[None, :, :]          # (B, ctx, d_model)
        q = torch.einsum("bcd,hde->bhce", x, self.W_Q)
        k = torch.einsum("bcd,hde->bhce", x, self.W_K)
        v = torch.einsum("bcd,hde->bhce", x, self.W_V)
        attn = torch.einsum("bhce,bhfe->bhcf", q, k) / math.sqrt(self.d_head)
        attn = attn.masked_fill(self.mask[None, None] == 0, -1e9).softmax(dim=-1)
        z = torch.einsum("bhcf,bhfe->bhce", attn, v)
        x = x + torch.einsum("bhce,hed->bcd", z, self.W_O)     # attention residual
        h = F.relu(x @ self.W_in + self.b_in)
        x = x + (h @ self.W_out + self.b_out)                  # MLP residual
        return x @ self.W_U                                    # (B, ctx, d_vocab_out)


def make_modular_addition_data(p: int):
    """All p^2 pairs as tokens [a, b, '='] -> target (a+b) mod p. '=' token id = p."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    X = torch.stack([a, b, eq], dim=1)          # (p^2, 3)
    Y = (a + b) % p                              # (p^2,)
    return X, Y
