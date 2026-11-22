"""Validated architecture specification; defaults have 120,054,528 parameters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16000
    context_length: int = 1024
    num_layers: int = 16
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    d_ff: int = 2240
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    user_token_id: int = 4
    assistant_token_id: int = 5

    def __post_init__(self):
        for name in ('vocab_size', 'context_length', 'num_layers', 'd_model',
                     'num_heads', 'num_kv_heads', 'd_ff'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        if self.num_heads % self.num_kv_heads:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        if self.head_dim % 2:
            raise ValueError('RoPE requires an even head_dim')
        for name in ('dropout', 'rope_theta', 'norm_eps', 'init_std'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if name == 'dropout':
                if not 0 <= value < 1:
                    raise ValueError('dropout must be in [0, 1)')
            elif value <= 0:
                raise ValueError(f'{name} must be positive')
        token_ids = [getattr(self, f'{role}_token_id') for role in
                     ('pad', 'bos', 'eos', 'unk', 'user', 'assistant')]
        if any(type(value) is not int or not 0 <= value < self.vocab_size for value in token_ids):
            raise ValueError('special token IDs must be integers in the vocabulary')
        if len(set(token_ids)) != len(token_ids):
            raise ValueError('special token IDs must be distinct')

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def pad_id(self):
        return self.pad_token_id

    @property
    def bos_id(self):
        return self.bos_token_id

    @property
    def eos_id(self):
        return self.eos_token_id

    @property
    def unk_id(self):
        return self.unk_token_id

    @property
    def user_id(self):
        return self.user_token_id

    @property
    def assistant_id(self):
        return self.assistant_token_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelConfig:
        if not isinstance(values, Mapping):
            raise ValueError('model config must be a mapping')
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown model config keys: {sorted(unknown)}')
        return cls(**values)
