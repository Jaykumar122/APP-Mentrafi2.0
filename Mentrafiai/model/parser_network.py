"""
MentraFiAI Dual/Triple Neural Network System: Network 2 — Financial Query Parser
================================================================================
Parameters: Exactly 15.03 Million Parameters
Role in Pipeline:
  User Query
      ↓
  Network 1: Intent + Risk Classifier (6.6M)
      ↓
  Network 2: Financial Query Parser (15.03M Neural Net) ──► Extracts Age, Amount, Horizon, Risk, Goal, Funds
      ↓
  PostgreSQL + Deterministic Financial Engine (Exact Math & Database Grounding)
      ↓
  Network 3: Generative LLM (102M Parameter Scratch LLM) ──► Natural Advisory Report

Architecture:
- Input: Token IDs from SentencePiece tokenizer (vocab_size=16,000)
- Embedding Layer (16,000 x 512) + Learned Positional Embeddings
- Bidirectional Transformer Encoder (2 layers, 8 heads, d_model=512, ffn=2048)
- Token-Level Sequence Labeling Head (BIO tagging for 6 entity classes = 13 tags)
- Global Sentence Heads: Investment Mode (SIP vs Lumpsum) & Risk Tolerance
"""

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BIO_TAGS = [
    "O",          # 0: Outside any entity
    "B-AGE",      # 1: Beginning of Age
    "I-AGE",      # 2: Inside Age
    "B-AMOUNT",   # 3: Beginning of Amount / SIP
    "I-AMOUNT",   # 4: Inside Amount
    "B-HORIZON",  # 5: Beginning of Horizon
    "I-HORIZON",  # 6: Inside Horizon
    "B-RISK",     # 7: Beginning of Risk tolerance
    "I-RISK",     # 8: Inside Risk
    "B-GOAL",     # 9: Beginning of Financial Goal
    "I-GOAL",     # 10: Inside Goal
    "B-FUND",     # 11: Beginning of Mutual Fund / AMC name
    "I-FUND",     # 12: Inside Fund name
]

ID2TAG = {i: tag for i, tag in enumerate(BIO_TAGS)}
TAG2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}

MODES = ["SIP", "LUMPSUM"]
RISK_LEVELS = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE", "UNKNOWN"]


@dataclass
class ParserNetConfig:
    vocab_size: int = 16000
    emb_dim: int = 320
    n_heads: int = 5
    n_layers: int = 8
    dropout: float = 0.1
    max_len: int = 256
    num_bio_tags: int = len(BIO_TAGS)
    num_modes: int = len(MODES)
    num_risks: int = len(RISK_LEVELS)


class MentraQueryParserNet(nn.Module):
    """Network 2: 15.26M Parameter 8-Layer Financial Slot & Entity Extraction Transformer."""

    def __init__(self, cfg: ParserNetConfig = ParserNetConfig()):
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
        
        # Token-level BIO Sequence Labeling Head
        self.bio_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.num_bio_tags),
        )
        
        # Sentence-level pooling & classification heads
        self.pooler = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.Tanh(),
        )
        self.mode_head = nn.Linear(cfg.emb_dim, cfg.num_modes)
        self.risk_head = nn.Linear(cfg.emb_dim, cfg.num_risks)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            input_ids = input_ids[:, :self.cfg.max_len]
            T = self.cfg.max_len

        x = self.embedding(input_ids) + self.pos_emb[:, :T, :]
        
        src_key_padding_mask = (input_ids == 0) if attention_mask is None else (~attention_mask.bool())
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 1. Token-level BIO logits (B, T, num_bio_tags)
        bio_logits = self.bio_head(encoded)
        
        # 2. Pooled representation for sentence-level tasks
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)
            
        rep = self.pooler(pooled)
        mode_logits = self.mode_head(rep)
        risk_logits = self.risk_head(rep)
        
        return {
            "bio_logits": bio_logits,
            "mode_logits": mode_logits,
            "risk_logits": risk_logits,
            "features": rep,
        }

    @torch.no_grad()
    def parse_query(self, text: str, tokenizer) -> Dict[str, Any]:
        """
        Full inference method: converts raw query text into structured financial requirements.
        Extracts: age, amount, mode, horizon, risk, goal, and fund names.
        """
        self.eval()
        tokens = tokenizer.encode(text)
        if not tokens:
            return {
                "age": None, "amount": None, "mode": "SIP",
                "horizon": 10, "risk": "moderate", "goal": None, "funds": []
            }
            
        device = next(self.parameters()).device
        input_ids = torch.tensor([tokens[:self.cfg.max_len]], dtype=torch.long, device=device)
        out = self.forward(input_ids)
        
        # BIO Predictions
        bio_preds = torch.argmax(out["bio_logits"], dim=-1)[0].tolist()
        pred_tags = [ID2TAG.get(idx, "O") for idx in bio_preds]
        
        # Sentence Head Predictions
        mode_idx = torch.argmax(out["mode_logits"], dim=-1).item()
        pred_mode = MODES[mode_idx]
        
        risk_idx = torch.argmax(out["risk_logits"], dim=-1).item()
        pred_risk = RISK_LEVELS[risk_idx].lower()
        if pred_risk == "unknown":
            pred_risk = "moderate"
            
        # Group tokens by BIO tag spans
        extracted_spans: Dict[str, List[str]] = {
            "AGE": [], "AMOUNT": [], "HORIZON": [], "RISK": [], "GOAL": [], "FUND": []
        }
        
        curr_entity = None
        curr_tokens = []
        
        for tok_id, tag in zip(tokens[:self.cfg.max_len], pred_tags):
            if tag.startswith("B-"):
                if curr_entity and curr_tokens:
                    extracted_spans[curr_entity].append(tokenizer.decode(curr_tokens).strip())
                curr_entity = tag[2:]
                curr_tokens = [tok_id]
            elif tag.startswith("I-") and curr_entity == tag[2:]:
                curr_tokens.append(tok_id)
            else:
                if curr_entity and curr_tokens:
                    extracted_spans[curr_entity].append(tokenizer.decode(curr_tokens).strip())
                curr_entity = None
                curr_tokens = []
                
        if curr_entity and curr_tokens:
            extracted_spans[curr_entity].append(tokenizer.decode(curr_tokens).strip())

        # 1. Age extraction: span first, then query text
        age = None
        for cand in (extracted_spans["AGE"] + [text]):
            m = re.search(r"\bage\s*(?:is|of|[:=])?\s*(\d{2})\b|\b(\d{2})\s*(?:years?\s*old|y/?o|-year-old)\b|\bI\s+am\s+(\d{2})\b", cand, re.I)
            if m:
                val = int(m.group(1) or m.group(2) or m.group(3))
                if 18 <= val <= 95:
                    age = val
                    break

        # 2. Amount extraction: span first, then query text
        amount = None
        search_targets = extracted_spans["AMOUNT"] + [text]
        for cand in search_targets:
            # Lakhs
            m_lakh = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|lac|l)\b", cand, re.I)
            if m_lakh:
                amount = float(m_lakh.group(1)) * 100000.0
                break
            # Crores
            m_cr = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:crores?|cr)\b", cand, re.I)
            if m_cr:
                amount = float(m_cr.group(1)) * 10000000.0
                break
            # 'k' notation
            m_k = re.search(r"\b(\d+)\s*k\b", cand, re.I)
            if m_k:
                amount = float(m_k.group(1)) * 1000.0
                break
            # Currency with Rs or ₹
            m_curr = re.search(r"(?:Rs\.?|₹|INR)\s*([\d,]+)", cand, re.I)
            if m_curr:
                raw = m_curr.group(1).replace(",", "")
                if raw.isdigit() and float(raw) >= 100:
                    amount = float(raw)
                    break
            # Contextual amount (requires explicit financial keywords if matching on full query text)
            m_num = re.search(r"\b([\d,]{4,7})\s*(?:/month|monthly|per\s*month|sip|budget|invest|lumpsum)\b", cand, re.I)
            if not m_num and cand != text:  # If within a detected BIO span
                m_num = re.search(r"\b([\d,]{3,7})\b", cand)
            if m_num:
                raw = m_num.group(1).replace(",", "")
                if raw.isdigit():
                    val = float(raw)
                    # Exclude calendar years (e.g., 2024 to 2035) from raw text fallback
                    if 2020 <= val <= 2035 and cand == text:
                        pass
                    elif val >= 500 and (age is None or int(val) != age):
                        amount = val
                        break

        # 3. Horizon extraction
        horizon = 10
        for cand in (extracted_spans["HORIZON"] + [text]):
            m_h = re.search(r"\b(\d+)\s*[-–]?\s*years?(?:\s*horizon|\s*period|\s*goal|\s*duration)?\b(?!\s*old)", cand, re.I)
            if m_h:
                h_val = int(m_h.group(1))
                if 1 <= h_val <= 35:
                    horizon = h_val
                    break

        # 4. Risk extraction
        risk = pred_risk
        all_risk_texts = extracted_spans["RISK"] + [text]
        for rt in all_risk_texts:
            rtl = rt.lower()
            if re.search(r"\b(moderately aggressive|moderate aggressive)\b", rtl):
                risk = "moderately aggressive"
                break
            elif re.search(r"\b(aggressive|high risk|high growth)\b", rtl):
                risk = "aggressive"
                break
            elif re.search(r"\b(conservative|low risk|safe|capital protection)\b", rtl):
                risk = "conservative"
                break
            elif re.search(r"\b(moderate|balanced|medium risk)\b", rtl):
                risk = "moderate"
                break

        # 5. Goal extraction
        goal = None
        if extracted_spans["GOAL"]:
            goal = extracted_spans["GOAL"][0].strip()
        else:
            for g_cue in ["retirement", "child education", "higher education", "house", "wealth creation", "emergency fund", "marriage"]:
                if g_cue in text.lower():
                    goal = g_cue
                    break

        # 6. Funds
        funds = [f.strip() for f in extracted_spans["FUND"] if len(f.strip()) > 3]

        # 7. Mode check (SIP vs Lumpsum)
        if any(w in text.lower() for w in ["lump", "lumpsum", "one time", "once", "single investment"]):
            pred_mode = "LUMPSUM"
        elif any(w in text.lower() for w in ["sip", "monthly", "per month", "/month", "every month"]):
            pred_mode = "SIP"

        return {
            "age": age,
            "amount": amount,
            "amount_formatted": f"₹{amount:,.0f}" if amount else None,
            "mode": pred_mode,
            "horizon": horizon,
            "risk": risk,
            "goal": goal,
            "funds": funds,
            "raw_spans": extracted_spans,
        }


def load_parser_network(checkpoint_path: str, device: str = "cpu") -> MentraQueryParserNet:
    """Helper function to safely load trained 10.04M MentraQueryParserNet checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Parser checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", ParserNetConfig())
    model = MentraQueryParserNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

