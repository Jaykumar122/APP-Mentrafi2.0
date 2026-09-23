"""
MentraFiAI Multi-Network System: Network 1 — Neural Intent & Parameter Classifier (10.16M)
==========================================================================================
Parameters: Exactly 10.16 Million Parameters
Role in Pipeline:
  User Query
      ↓
  Network 1: Intent + Risk Classifier (10.16M) ──► Intent, Risk Appetite, Investment Horizon
      ↓
  Network 2: Financial Query Parser (15.03M) ──► Age, Amount, Mode, Horizon, Goal, Funds
      ↓
  PostgreSQL + Deterministic Financial Engine (Exact Math & Database Grounding)
      ↓
  Network 3: Generative LLM (102.5M) ──► Natural Advisory Report

Architecture:
- Input: Token IDs from SentencePiece tokenizer (vocab_size=16,000)
- Embedding Layer (16,000 x 384) + Learned Positional Embeddings
- Bidirectional Transformer Encoder (2 layers, 6 heads, d_model=384, ffn=1536)
- Tanh Representation Pooling
- Multiple Classification Heads:
    1. Intent Head (Greeting, Definition, Portfolio, Comparison, Goal, General)
    2. Risk Appetite Head (Conservative, Moderate, Aggressive, Unknown)
    3. Horizon Head (Short <3y, Medium 3-7y, Long 7y+, Unknown)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any


INTENTS = [
    "GREETING",
    "DEFINITION",
    "PORTFOLIO_RECOMMENDATION",
    "FUND_COMPARISON",
    "GOAL_PLANNING",
    "GENERAL_CHAT"
]

RISK_CLASSES = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE", "UNKNOWN"]
HORIZON_CLASSES = ["SHORT_TERM", "MEDIUM_TERM", "LONG_TERM", "UNKNOWN"]


@dataclass
class ClassifierConfig:
    vocab_size: int = 16000
    emb_dim: int = 256
    n_heads: int = 4
    n_layers: int = 8
    num_intents: int = len(INTENTS)
    num_risk: int = len(RISK_CLASSES)
    num_horizons: int = len(HORIZON_CLASSES)
    dropout: float = 0.1
    max_len: int = 256


class MentraIntentClassifier(nn.Module):
    """Neural Network for Query Intent & Financial Parameter Detection."""

    def __init__(self, cfg: ClassifierConfig = ClassifierConfig()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_len, cfg.emb_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.emb_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.pooler = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.Tanh()
        )
        
        # Classification Heads
        self.intent_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.emb_dim, cfg.emb_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.emb_dim // 2, cfg.num_intents)
        )
        
        self.risk_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.emb_dim, cfg.emb_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.emb_dim // 2, cfg.num_risk)
        )

        self.horizon_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.emb_dim, cfg.emb_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.emb_dim // 2, cfg.num_horizons)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            input_ids = input_ids[:, :self.cfg.max_len]
            T = self.cfg.max_len

        x = self.embedding(input_ids) + self.pos_emb[:, :T, :]
        
        src_key_padding_mask = (input_ids == 0) if attention_mask is None else (~attention_mask.bool())
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Masked average pooling
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)

        rep = self.pooler(pooled)

        return {
            "intent_logits": self.intent_head(rep),
            "risk_logits": self.risk_head(rep),
            "horizon_logits": self.horizon_head(rep),
            "embedding": rep,
        }

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Inference helper for downstream router and generation controllers."""
        self.eval()
        out = self.forward(input_ids)
        intent_idx = torch.argmax(out["intent_logits"], dim=-1).item()
        risk_idx = torch.argmax(out["risk_logits"], dim=-1).item()
        horizon_idx = torch.argmax(out["horizon_logits"], dim=-1).item()

        return {
            "intent": INTENTS[intent_idx],
            "intent_confidence": F.softmax(out["intent_logits"], dim=-1)[0, intent_idx].item(),
            "predicted_risk": RISK_CLASSES[risk_idx],
            "predicted_horizon": HORIZON_CLASSES[horizon_idx],
        }
