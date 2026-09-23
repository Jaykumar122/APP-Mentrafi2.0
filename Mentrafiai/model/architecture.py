"""
MentraFiAI model architecture.

A minimal, from-scratch, Claude/Llama-style decoder-only transformer:
  - Token embeddings (no learned absolute positions — RoPE handles position)
  - N x TransformerBlock:
        RMSNorm -> Causal Self-Attention (RoPE + Grouped-Query Attention) -> residual
        RMSNorm -> SwiGLU MLP -> residual
  - Final RMSNorm -> Linear head, weight-tied to the token embedding

No HuggingFace `transformers` dependency — everything below is implemented directly
so the internals are fully visible and educational.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm — used instead of LayerNorm. Simpler (no mean-subtraction / bias),
# cheaper, and what modern Llama/Claude-style models use.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize by root-mean-square instead of mean/variance
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — encodes position by rotating pairs of
# dimensions in the Q/K vectors, rather than adding a learned position vector.
# This generalizes better to longer sequences than learned absolute positions.
# ---------------------------------------------------------------------------
def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim)
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    # Rotate: standard RoPE pairwise rotation
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA) with RoPE and a causal mask.
# n_kv_head < n_head: multiple query heads share the same K/V head, which
# cuts the KV cache / projection cost substantially vs full multi-head attention.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head  # how many Q heads share one KV head
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads (GQA repeat)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Fused, causal scaled-dot-product attention (memory-efficient on T4)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return self.resid_dropout(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP — gated feed-forward used in Llama/PaLM-style models. Empirically
# outperforms plain GELU MLPs at matched parameter count.
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.hidden_dim
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MentraFiAI(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)

        # Weight-tied output head (saves vocab_size * n_embd params)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight

        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Off by default. Pretrain enables this for the 150M model so the GTX 1650
        # stays under the ~3 GB WDDM spill cliff (plain 150M peak was 3.07 GB).
        self.gradient_checkpointing = False

        self.apply(self._init_weights)
        # Scale down residual-path projections (o_proj, down_proj) by 1/sqrt(2*n_layer).
        # Done in a single pass AFTER apply() to avoid O(n^2) re-scan.
        # Prevents gradient explosion accumulating across layers — standard in Llama/GPT-NeoX.
        residual_std = self.cfg.init_std / (2 * self.cfg.n_layer) ** 0.5
        for name, child in self.named_modules():
            if isinstance(child, nn.Linear) and name.endswith(("o_proj", "down_proj")):
                nn.init.normal_(child.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embed.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.dropout(self.tok_embed(idx))
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        x = self.final_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.9,
                 repetition_penalty=1.3, no_repeat_ngram_size=3):
        """Autoregressive sampling with anti-degeneration guards.

        A ~100M model at low temperature reliably falls into repeat loops and
        often never emits EOS, so on top of top-k/top-p we apply:
          * repetition_penalty (CTRL-style): logits of already-generated tokens
            are divided by the penalty (>1 discourages, keeps sign correct).
          * no_repeat_ngram_size: any token that would complete a previously seen
            n-gram is masked to -inf so exact n-gram loops can't form.
        Both default on; pass repetition_penalty=1.0 / no_repeat_ngram_size=0 to
        recover the plain sampler.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Repetition penalty over tokens already in the sequence (per batch row).
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    row = logits[b, seen]
                    logits[b, seen] = torch.where(row > 0, row / repetition_penalty,
                                                  row * repetition_penalty)

            # Block tokens that would complete a repeated n-gram.
            if no_repeat_ngram_size and idx.size(1) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for b in range(idx.size(0)):
                    seq = idx[b].tolist()
                    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                    banned = set()
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i:i + n - 1]) == prefix:
                            banned.add(seq[i + n - 1])
                    for tok in banned:
                        logits[b, tok] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            if next_token.item() == self.cfg.eos_token_id:
                break
        return idx
