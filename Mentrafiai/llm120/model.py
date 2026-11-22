"""A decoder implemented directly in PyTorch, without pretrained model code.

Labels are ALREADY next-token shifted by the datasets. Only -100 is ignored.
KV caches store rotated, unexpanded keys and values; cache offsets participate
in both rotary positions and causal masking (including multi-token decode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    num_targets: int = 0
    past_key_values: tuple[KVCache, ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(x.dtype)


def rotate(x, cos, sin):
    """Split-half RoPE; precomputed tables are sliced at the cache offset."""
    left, right = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        kv_width = config.num_kv_heads * config.head_dim
        self.k = nn.Linear(config.d_model, kv_width, bias=False)
        self.v = nn.Linear(config.d_model, kv_width, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        batch, length, width = x.shape
        config = self.config
        q = self.q(x).view(batch, length, config.num_heads, config.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, config.num_kv_heads, config.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, config.num_kv_heads, config.head_dim).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        offset = 0
        if past is not None:
            offset = past[0].shape[2]
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present = (k, v) if use_cache else None
        repeats = config.num_heads // config.num_kv_heads
        # Explicit expansion supports PyTorch 2.1 and the T4 SDPA backends.
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        mask = None
        if offset and length > 1:
            query_positions = torch.arange(offset, offset + length, device=x.device)
            key_positions = torch.arange(offset + length, device=x.device)
            mask = key_positions[None, :] <= query_positions[:, None]
        values = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=config.dropout if self.training else 0.0,
            is_causal=offset == 0,
        )
        values = values.transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.out(values)), present


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        attention, present = self.attention(self.attention_norm(x), cos, sin, past, use_cache)
        x = x + attention
        normalized = self.ffn_norm(x)
        return x + self.dropout(self.down(F.silu(self.gate(normalized)) * self.up(normalized))), present


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.dropout = nn.Dropout(config.dropout)
        frequency = config.rope_theta ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        angles = torch.outer(torch.arange(config.context_length).float(), frequency)
        self.register_buffer('rope_cos', angles.cos(), persistent=False)
        self.register_buffer('rope_sin', angles.sin(), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and module is not self.lm_head:
                nn.init.normal_(module.weight, std=config.init_std)
        residual_std = config.init_std / math.sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out.weight, std=residual_std)
            nn.init.normal_(block.down.weight, std=residual_std)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids, labels=None, *, use_cache=False, past_key_values=None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError('input_ids must have shape [batch, nonempty sequence]')
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError('labels must match input_ids and already be next-token shifted')
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError('KV caching is for evaluation only')
        if past_key_values is not None and not use_cache:
            raise ValueError('past_key_values requires use_cache=True')
        offset = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks):
                raise ValueError('KV cache must contain one entry per layer')
            lengths = []
            for key, value in past_key_values:
                expected = (input_ids.shape[0], self.config.num_kv_heads)
                if (key.ndim != 4 or key.shape != value.shape or key.shape[:2] != expected
                        or key.shape[-1] != self.config.head_dim or key.device != input_ids.device):
                    raise ValueError('KV cache shape/device does not match input/model')
                lengths.append(key.shape[2])
            if len(set(lengths)) != 1:
                raise ValueError('KV cache lengths must agree across layers')
            offset = lengths[0]
        length = input_ids.shape[1]
        if offset + length > self.config.context_length:
            raise ValueError('Sequence plus cache exceeds context_length; shorten the prompt')
        x = self.dropout(self.embedding(input_ids))
        cos = self.rope_cos[offset:offset + length]
        sin = self.rope_sin[offset:offset + length]
        caches = []
        for index, block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                # Bind each block to avoid late-binding the final layer in backward.
                def run(hidden, cosine, sine, layer=block):
                    return layer(hidden, cosine, sine)[0]
                x = checkpoint(run, x, cos, sin, use_reentrant=False)
            else:
                x, present = block(x, cos, sin,
                                   None if past_key_values is None else past_key_values[index], use_cache)
                if use_cache:
                    caches.append(present)
        logits = self.lm_head(self.norm(x))
        result = LMOutput(logits=logits, past_key_values=tuple(caches) if use_cache else None)
        if labels is not None:
            result.num_targets = int((labels != -100).sum().item())
            if not result.num_targets:
                raise ValueError('Batch contains no supervised targets (all labels are -100)')
            # Chunk the FP32 CE conversion to avoid a second full FP32 logits tensor.
            flat_logits, flat_labels = logits.reshape(-1, self.config.vocab_size), labels.reshape(-1)
            total = logits.new_zeros((), dtype=torch.float32)
            for start in range(0, flat_labels.numel(), 128):
                total = total + F.cross_entropy(flat_logits[start:start + 128].float(),
                                               flat_labels[start:start + 128],
                                               ignore_index=-100, reduction='sum')
            result.loss_sum = total
            result.loss = total / result.num_targets
        return result
