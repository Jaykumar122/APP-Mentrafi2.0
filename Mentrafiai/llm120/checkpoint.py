"""Atomic, self-describing checkpoints with strict tokenizer/template identity.

Only load checkpoints you trust. weights_only=True avoids arbitrary pickle
objects; it does not make untrusted tensor archives a resource-safe input.
Legacy .pt files are intentionally not accepted by this pipeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import CausalLM
from .tokenization import TEMPLATE_VERSION, Tokenizer

FORMAT_VERSION = 'mentrafiai-llm120-v1'


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Unsupported checkpoint object: {type(value).__name__}')


def _check_weights(state):
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint has no model_state')
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f'Model state {name} is not a tensor')
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite checkpoint weight: {name}; weights were not repaired')
    if ('embedding.weight' not in state or 'lm_head.weight' not in state
            or not torch.equal(state['embedding.weight'], state['lm_head.weight'])):
        raise ValueError('Checkpoint must contain identical tied embedding/head weights')


def save_checkpoint(path, model: CausalLM, tokenizer: Tokenizer, *, optimizer=None,
                    scaler=None, training_state=None, metadata=None) -> None:
    tokenizer.validate_config(model.config)
    # All copies must succeed. A failed GPU transfer must NEVER become zero weights.
    model_state = _cpu_copy(model.state_dict())
    _check_weights(model_state)
    payload = {
        'format_version': FORMAT_VERSION,
        'model_config': model.config.to_dict(),
        'model_state': model_state,
        'tokenizer_sha256': tokenizer.sha256,
        'template_version': TEMPLATE_VERSION,
        'optimizer': _cpu_copy(optimizer.state_dict()) if optimizer is not None else None,
        'scaler': _cpu_copy(scaler.state_dict()) if scaler is not None else None,
        'training_state': _cpu_copy(training_state),
        'metadata': _cpu_copy(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_weights(source, destination) -> Path:
    """Write a serving copy without optimizer/scaler/training state (about 1/3 the size).

    The result loads with load_checkpoint (serve.py) but cannot be used for --resume.
    """
    payload = torch.load(Path(source), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Not an llm120 v1 checkpoint')
    _check_weights(payload['model_state'])
    slim = {**payload, 'optimizer': None, 'scaler': None, 'training_state': None}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(slim, stream)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_checkpoint(path, tokenizer: Tokenizer, device='cpu') -> tuple[CausalLM, dict[str, Any]]:
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Not an llm120 v1 checkpoint; legacy model weights require a separate pipeline')
    if payload.get('tokenizer_sha256') != tokenizer.sha256:
        raise ValueError('Checkpoint tokenizer_sha256 mismatch; use the exact training tokenizer')
    if payload.get('template_version') != TEMPLATE_VERSION:
        raise ValueError('Checkpoint chat template mismatch')
    required = {'model_config', 'model_state', 'optimizer', 'scaler', 'training_state', 'metadata'}
    if not required.issubset(payload):
        raise ValueError(f'Incomplete checkpoint; missing {sorted(required - set(payload))}')
    config = ModelConfig.from_dict(payload['model_config'])
    tokenizer.validate_config(config)
    _check_weights(payload['model_state'])
    model = CausalLM(config)
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device).eval()
    return model, payload
