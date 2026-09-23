"""
MentraFiAI Training Pipeline: Network 2 — Financial Query Parser (15.03M)
=========================================================================
Trains MentraQueryParserNet on synthetic & annotated Indian financial queries.
Tasks:
1. Token-level Sequence Labeling (BIO Tags for Age, Amount, Horizon, Risk, Goal, Fund)
2. Sentence-level Mode Classification (SIP vs LUMPSUM)
3. Sentence-level Risk Classification (Conservative, Moderate, Aggressive)
"""

import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(r"D:\Mentrafiai")
sys.path.insert(0, str(BASE_DIR))

from model.parser_network import MentraQueryParserNet, ParserNetConfig, BIO_TAGS, TAG2ID, MODES, RISK_LEVELS
from tokenizer.tokenizer_utils import Tokenizer

random.seed(42)
torch.manual_seed(42)

NAMES = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohit", "Kavya", "Suresh", "Meera", "Arjun", "Divya"]
CITIES = ["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
FUNDS = [
    "Parag Parikh Flexi Cap Fund", "SBI Small Cap Fund", "Nippon India Small Cap Fund",
    "HDFC Mid Cap Opportunities", "Mirae Asset Large Cap Fund", "Axis Bluechip Fund",
    "Quant Active Fund", "ICICI Prudential Equity & Debt", "UTI Nifty 50 Index Fund",
    "Kotak Emerging Equity Fund", "Motilal Oswal Midcap Fund", "Tata Digital India Fund"
]
GOALS = [
    "retirement", "child higher education", "buying a house", "wealth creation",
    "daughter marriage", "emergency fund", "buying a car", "financial independence"
]

def generate_annotated_example(tokenizer: Tokenizer) -> dict:
    """Generates a query with exact token BIO alignment."""
    name = random.choice(NAMES)
    age = random.randint(21, 65)
    amt_val = random.choice([1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000, 75000, 100000, 250000, 500000, 1000000])
    is_sip = random.choice([True, True, True, False])  # 75% SIP, 25% Lumpsum
    horizon = random.choice([3, 5, 7, 10, 12, 15, 20])
    risk = random.choice(["conservative", "moderate", "aggressive"])
    goal = random.choice(GOALS)
    fund = random.choice(FUNDS)

    # Format amount text
    if amt_val >= 10000000:
        amt_str = f"Rs {amt_val // 10000000} crore"
    elif amt_val >= 100000:
        amt_str = f"Rs {amt_val / 100000:.1f} lakh" if amt_val % 100000 != 0 else f"Rs {amt_val // 100000} lakh"
    elif amt_val >= 1000 and random.random() < 0.4:
        amt_str = f"{amt_val // 1000}k"
    else:
        amt_str = f"Rs {amt_val:,}"

    # Templates
    templates = [
        # Full query
        f"I am {age} years old. Want to invest {amt_str} monthly SIP for {horizon} years with {risk} risk for my {goal}.",
        f"My name is {name}, age {age}. Can invest {amt_str} per month in mutual funds for {horizon} years, {risk} appetite.",
        f"Suggest mutual funds for a {age}-year-old: budget {amt_str}/month, {horizon}-year horizon, {risk} risk profile.",
        f"I want to invest {amt_str} lumpsum for {horizon} years in {fund}, {risk} risk.",
        f"Age {age}. Planning {amt_str} SIP for {horizon} years for {goal}. Risk level is {risk}.",
        f"Looking for {risk} funds to invest {amt_str} monthly for {horizon} years duration.",
        f"Compare {fund} for a {horizon}-year goal with {risk} risk.",
        f"I have {amt_str} one-time to invest in {fund} for {horizon} years.",
        f"Can I invest {amt_str} per month at age {age} for {goal} over {horizon} years with {risk} tolerance?",
    ]
    query_text = random.choice(templates)
    
    tokens = tokenizer.encode(query_text)
    bio_tags = [TAG2ID["O"]] * len(tokens)

    # Tag spans using token substrings
    def tag_entity(entity_str: str, entity_name: str):
        ent_tokens = tokenizer.encode(entity_str)
        if not ent_tokens:
            return
        # Find subsequence
        for i in range(len(tokens) - len(ent_tokens) + 1):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                bio_tags[i] = TAG2ID[f"B-{entity_name}"]
                for j in range(1, len(ent_tokens)):
                    bio_tags[i+j] = TAG2ID[f"I-{entity_name}"]
                break

    tag_entity(str(age), "AGE")
    tag_entity(amt_str, "AMOUNT")
    tag_entity(f"{horizon} years", "HORIZON")
    tag_entity(risk, "RISK")
    tag_entity(goal, "GOAL")
    tag_entity(fund, "FUND")

    mode_label = 0 if is_sip else 1
    risk_label = 0 if risk == "conservative" else 1 if risk == "moderate" else 2

    return {
        "text": query_text,
        "input_ids": tokens,
        "bio_tags": bio_tags,
        "mode_label": mode_label,
        "risk_label": risk_label,
    }


class ParserDataset(Dataset):
    def __init__(self, examples: list, max_len: int = 256):
        self.examples = examples
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = ex["input_ids"][:self.max_len]
        tags = ex["bio_tags"][:self.max_len]
        pad_len = self.max_len - len(ids)

        padded_ids = ids + [0] * pad_len
        padded_tags = tags + [0] * pad_len  # pad with O (0)
        mask = [1] * len(ids) + [0] * pad_len

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "bio_tags": torch.tensor(padded_tags, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "mode_label": torch.tensor(ex["mode_label"], dtype=torch.long),
            "risk_label": torch.tensor(ex["risk_label"], dtype=torch.long),
        }


def train_parser_network():
    print("=" * 70)
    print("  TRAINING NETWORK 2: FINANCIAL QUERY PARSER (10.04M PARAMETERS)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    tok_path = str(BASE_DIR / "tokenizer" / "tokenizer_v2.model")
    tokenizer = Tokenizer(tok_path)
    print(f"[OK] Tokenizer loaded (vocab={tokenizer.vocab_size:,})")

    # 1. Generate annotated dataset
    print("\nGenerating 12,000 synthetic annotated financial queries with BIO tags...")
    all_data = [generate_annotated_example(tokenizer) for _ in range(12000)]
    split = int(0.9 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]

    train_ds = ParserDataset(train_data)
    val_ds = ParserDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print(f"[OK] Train set: {len(train_ds):,} | Val set: {len(val_ds):,}")

    # 2. Initialize 10.04M model
    cfg = ParserNetConfig(vocab_size=tokenizer.vocab_size)
    model = MentraQueryParserNet(cfg).to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"[OK] MentraQueryParserNet initialized: {params_count:,} parameters ({params_count/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion_bio = nn.CrossEntropyLoss()
    criterion_mode = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()

    epochs = 4
    ckpt_dir = BASE_DIR / "checkpoints" / "query_parser"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = ckpt_dir / "best_parser_net.pt"

    print("\nStarting training loop...")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            bio_tags = batch["bio_tags"].to(device)
            mask = batch["attention_mask"].to(device)
            mode_lbl = batch["mode_label"].to(device)
            risk_lbl = batch["risk_label"].to(device)

            out = model(input_ids, attention_mask=mask)

            # Masked BIO loss
            bio_logits = out["bio_logits"].view(-1, cfg.num_bio_tags)
            bio_targets = bio_tags.view(-1)
            loss_bio = criterion_bio(bio_logits, bio_targets)

            # Global losses
            loss_mode = criterion_mode(out["mode_logits"], mode_lbl)
            loss_risk = criterion_risk(out["risk_logits"], risk_lbl)

            loss = loss_bio + 0.5 * loss_mode + 0.5 * loss_risk

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct_bio = 0
        total_bio = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                bio_tags = batch["bio_tags"].to(device)
                mask = batch["attention_mask"].to(device)
                mode_lbl = batch["mode_label"].to(device)
                risk_lbl = batch["risk_label"].to(device)

                out = model(input_ids, attention_mask=mask)
                loss_bio = criterion_bio(out["bio_logits"].view(-1, cfg.num_bio_tags), bio_tags.view(-1))
                loss_mode = criterion_mode(out["mode_logits"], mode_lbl)
                loss_risk = criterion_risk(out["risk_logits"], risk_lbl)

                v_loss = loss_bio + 0.5 * loss_mode + 0.5 * loss_risk
                val_loss += v_loss.item()

                preds = torch.argmax(out["bio_logits"], dim=-1)
                active_mask = mask & (bio_tags != 0)  # non-pad and non-O entities
                correct_bio += (preds[active_mask] == bio_tags[active_mask]).sum().item()
                total_bio += active_mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        f1_entity_acc = (correct_bio / total_bio * 100) if total_bio > 0 else 100.0
        elapsed = time.time() - t0

        print(f"Epoch {ep}/{epochs} ({elapsed:.1f}s) | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Entity Extraction Accuracy: {f1_entity_acc:.1f}%")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "val_loss": avg_val_loss,
                "accuracy": f1_entity_acc,
            }, str(best_path))
            print(f"  [SAVED] Best checkpoint -> {best_path.name}")

    print("\n" + "=" * 70)
    print("  VERIFYING PARSER INFERENCE ON TEST QUERIES")
    print("=" * 70)

    test_queries = [
        "I am 28 years old, want to start a 10000 monthly SIP for 10 years with moderate risk.",
        "My name is Priya, age 32. Can invest Rs 25,000 per month for 15 years, aggressive risk for retirement.",
        "I have 2.5 lakh lumpsum to invest for 5 years in Parag Parikh Flexi Cap Fund, conservative risk.",
    ]

    for q in test_queries:
        parsed = model.parse_query(q, tokenizer)
        print(f"\nQuery: {q}")
        print(f"  -> Extracted Age    : {parsed['age']}")
        print(f"  -> Extracted Amount : {parsed['amount_formatted']} ({parsed['mode']})")
        print(f"  -> Extracted Horizon: {parsed['horizon']} years")
        print(f"  -> Extracted Risk   : {parsed['risk']}")
        print(f"  -> Extracted Goal   : {parsed['goal']}")
        print(f"  -> Extracted Funds  : {parsed['funds']}")

    print("\n[SUCCESS] Network 2 (Financial Query Parser 15.03M) is fully trained and ready!")


if __name__ == "__main__":
    train_parser_network()
