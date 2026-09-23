"""
Model hyperparameters for MentraFiAI (Claude/Llama-style decoder-only transformer).

Target: ~100M parameters, trainable on a single T4 (16GB) within a 30-40 GPU-hour budget.

Verified param count for n_layer=14, n_embd=768, n_head=12, n_kv_head=4,
vocab_size=16000, block_size=1024 (measured via model.num_parameters()):

    Total parameters:          102,455,040   (~102.5M)
    Non-embedding parameters:   90,166,784   (~90.2M)

To retarget: adjust n_layer/n_embd in configs/model_config.yaml, then check
model.num_parameters() directly before committing to a long training run,
rather than trusting hand-math alone.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "mentrafiai"

    vocab_size: int = 16000        # tuned to dataset size; SentencePiece BPE
    block_size: int = 1024         # context length (max_seq_length)

    n_layer: int = 14
    n_head: int = 12               # query heads
    n_kv_head: int = 4             # GQA: fewer KV heads than query heads (saves memory/compute)
    n_embd: int = 768

    mlp_ratio: float = 2.67        # SwiGLU hidden expansion (matches ~4x GELU param count)

    dropout: float = 0.0
    rope_theta: float = 10000.0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    user_token_id: int = 4
    assistant_token_id: int = 5

    init_std: float = 0.01

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def hidden_dim(self) -> int:
        h = int(self.mlp_ratio * self.n_embd)
        return (h + 63) // 64 * 64
