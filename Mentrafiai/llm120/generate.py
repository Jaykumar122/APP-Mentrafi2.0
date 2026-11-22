"""Raw model decoding: no canned answers, glossary routing, or silent fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .tokenization import serialize


@dataclass
class GenerationResult:
    token_ids: list[int]
    finish_reason: str


def sample_token(logits, *, temperature, top_k, top_p, generator=None):
    if not torch.isfinite(logits).any():
        raise FloatingPointError('No finite next-token logits')
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise FloatingPointError('Invalid next-token logits; inspect checkpoint/precision')
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    values = logits / temperature
    if top_k:
        threshold = torch.topk(values, min(top_k, values.shape[-1]), dim=-1).values[..., -1:]
        values = values.masked_fill(values < threshold, float('-inf'))
    if top_p < 1:
        ordered, indices = torch.sort(values, descending=True, dim=-1)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        # Include the first token crossing the threshold, including top_p near 0.
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, float('-inf'))
        values = torch.full_like(values, float('-inf')).scatter(-1, indices, ordered)
    return torch.multinomial(values.softmax(-1), 1, generator=generator)


@torch.inference_mode()
def generate_ids(model, input_ids, *, max_new_tokens=192, temperature=0.7, top_k=40,
                 top_p=0.9, repetition_penalty=1.05, use_cache=True, generator=None):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError('Generation supports one nonempty prompt at a time')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer (0 disables it)')
    for name, value in (('temperature', temperature), ('top_p', top_p),
                        ('repetition_penalty', repetition_penalty)):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if temperature < 0 or not 0 < top_p <= 1 or repetition_penalty < 1:
        raise ValueError('Require temperature >= 0, 0 < top_p <= 1, repetition_penalty >= 1')
    if input_ids.shape[1] + max_new_tokens > model.config.context_length:
        raise ValueError('Prompt + max_new_tokens exceeds context_length; shorten history/prompt or reduce max tokens')
    was_training = model.training
    model.eval()
    generated = []
    sequence = input_ids
    cache = None
    config = model.config
    blocked = [config.pad_token_id, config.bos_token_id, config.unk_token_id,
               config.user_token_id, config.assistant_token_id]
    try:
        for _ in range(max_new_tokens):
            current = sequence if cache is None else sequence[:, -1:]
            output = model(current, use_cache=use_cache, past_key_values=cache)
            cache = output.past_key_values if use_cache else None
            logits = output.logits[:, -1].float().clone()
            logits[:, blocked] = float('-inf')
            # Only penalize generated content, never grounded facts in the prompt.
            if generated and repetition_penalty != 1:
                seen = list(set(generated))
                values = logits[:, seen]
                logits[:, seen] = torch.where(values >= 0, values / repetition_penalty,
                                               values * repetition_penalty)
            token = sample_token(logits, temperature=temperature, top_k=top_k,
                                 top_p=top_p, generator=generator)
            value = int(token.item())
            if value == config.eos_token_id:
                return GenerationResult(generated, 'eos')
            generated.append(value)
            sequence = torch.cat((sequence, token), dim=1)
        return GenerationResult(generated, 'length')
    finally:
        model.train(was_training)


def generate_chat(model, tokenizer, messages, **kwargs):
    """Return (decoded answer, generation metadata), preserving model formatting."""
    tokenizer.validate_config(model.config)
    ids, _ = serialize(messages, tokenizer, add_generation_prompt=True)
    device = next(model.parameters()).device
    tensor = torch.tensor([ids], device=device, dtype=torch.long)
    result = generate_ids(model, tensor, **kwargs)
    return tokenizer.decode(result.token_ids), result
