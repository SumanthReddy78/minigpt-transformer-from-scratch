"""
Decoder-only Transformer, implemented from scratch in PyTorch.

This is the same family of architecture behind GPT-style models:
- Token + learned positional embeddings
- Multi-head causal self-attention (implemented manually, not via nn.MultiheadAttention,
  so every matrix multiply is visible and explainable in an interview)
- Position-wise feed-forward network
- Pre-norm residual blocks
- Weight-tied output head

Author: <you> — built as part of an ML Engineer portfolio project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, written explicitly."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined projection for Q, K, V (one matmul instead of three)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3 * embed_dim)
        q, k, v = qkv.split(embed_dim, dim=2)

        # Reshape to (B, num_heads, T, head_dim) for parallel head computation
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention: (B, heads, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: token i can only attend to tokens <= i
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        allowed = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Optional padding mask (1 = real token, 0 = pad): pad tokens must never
        # be attended to as keys, even by other real tokens under the causal mask.
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()  # (B, 1, 1, T)
            allowed = allowed & key_mask
            # A padded query position can end up with zero valid keys (every
            # key before and including it is also padding), which makes its
            # softmax row entirely -inf -> NaN (0/0). That row's own output is
            # never used, but the NaN still poisons every gradient that flows
            # back through the shared projection weights. Always permitting
            # self-attention guarantees at least one valid key per row, so no
            # row is ever fully masked out — this fixes both forward and
            # backward, whereas zeroing the output afterward only fixes forward.
            diagonal = torch.eye(seq_len, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            allowed = allowed | diagonal

        attn_scores = attn_scores.masked_fill(~allowed, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network (standard 4x expansion)."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> residual, LN -> FFN -> residual."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    A small decoder-only transformer language model.

    Default config (~10M params) trains on CPU in minutes on a small corpus —
    enough to demonstrate the architecture works, not to be state of the art.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: input embedding and output head share weights (standard trick,
        # cuts parameter count and tends to improve small-model performance)
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, idx: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Return final-layer hidden states (B, T, embed_dim) — shared by the
        language-modeling head and, when reused as a backbone, a classification head."""
        batch_size, seq_len = idx.shape
        assert seq_len <= self.block_size, "sequence longer than block_size"

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attention_mask)
            if attention_mask is not None:
                # Padded query positions have no valid keys at all (every key
                # is masked out), so softmax over an all -inf row yields NaN
                # for that position's own output. That NaN must be explicitly
                # overwritten (not multiplied by the mask — 0 * NaN is still
                # NaN in IEEE float) or it silently contaminates every real
                # position in every subsequent layer.
                pad_positions = (attention_mask == 0).unsqueeze(-1).expand_as(x)
                x = x.masked_fill(pad_positions, 0.0)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, attention_mask: torch.Tensor = None):
        x = self.encode(idx, attention_mask)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Autoregressive sampling, one token at a time."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
