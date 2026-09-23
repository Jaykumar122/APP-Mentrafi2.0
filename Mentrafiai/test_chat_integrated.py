"""
Test the FULL integrated chat.py system (not just the LLM)
This will test the complete pipeline with intent classification and grounded generation
"""

import sys
import os
from pathlib import Path
import torch
import re

if sys.platform == "win32":
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from model import ModelConfig, MentraFiAI
from tokenizer.tokenizer_utils import Tokenizer
from training.utils import load_config
from inference.generate import generate_chat_reply, is_greeting, is_informational, informational_reply, GREETING_REPLY
from inference.db_fund_retriever import FundDatabase
from chat import (
    extract_portfolio_params,
    generate_grounded_portfolio,
    has_severe_repetition,
    has_negative_amounts,
    has_nonsensical_numbers,
    validate_llm_response
)


def simulate_chat_interaction(query: str, model, tokenizer, device, classifier, db):
    """Simulate what chat.py would do with a query"""
    print(f"\n{'='*80}")
    print(f"USER QUERY: {query}")
    print(f"{'='*80}")

    # Step 1: Clean the input
    def clean_financial_prompt(text: str) -> str:
        text = text.strip("\"'""'' `")
        text = text.replace("₹", "Rs ")
        text = re.sub(r"\b(\d+)\s*[-–]?\s*years?\s*[-–]?\s*old\b", r"age \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\bI am (\d+)\b(?!\s*(?:lakh|crore|k|thousand))", r"age \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", lambda m: f"Rs {int(float(m.group(1))*10000000):,}", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", lambda m: f"Rs {int(float(m.group(1))*100000):,}", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    clean_input = clean_financial_prompt(query)
    print(f"[1] Cleaned: {clean_input}")

    # Step 2: Greeting check
    if is_greeting(clean_input):
        print(f"[2] Route: GREETING")
        print(f"\nRESPONSE:\n{GREETING_REPLY}")
        return

    # Step 3: Informational check
    if is_informational(clean_input):
        reply = informational_reply(clean_input)
        print(f"[2] Route: INFORMATIONAL/GLOSSARY")
        print(f"\nRESPONSE:\n{reply}")
        return

    # Step 4: Intent classification
    intent = "GENERAL"
    risk = "MODERATE"
    if classifier is not None:
        query_tokens = torch.tensor([tokenizer.encode(clean_input)[:128]], dtype=torch.long, device=device)
        pred = classifier.predict(query_tokens)
        intent = pred["intent"]
        risk = pred["predicted_risk"]
        print(f"[2] Intent Classifier: {intent} | Risk: {risk}")

    # Step 5: Extract parameters
    params = extract_portfolio_params(clean_input, risk)
    print(f"[3] Parameters: Budget={params['budget']}, Horizon={params['horizon']}, Risk={params['risk']}, Age={params.get('age')}")

    # Step 6: Route based on intent
    if intent == "PORTFOLIO_RECOMMENDATION" and params["budget"] is not None:
        print(f"[4] Route: GROUNDED PORTFOLIO GENERATOR")
        reply = generate_grounded_portfolio(
            budget=params["budget"],
            risk=params["risk"],
            horizon=params["horizon"],
            name=params["name"],
            age=params.get("age"),
            db=db
        )
        print(f"\nRESPONSE (first 500 chars):\n{reply[:500]}...")
        print(f"\n[✓] Response uses DETERMINISTIC calculations (no LLM hallucination)")
        return

    elif intent == "PORTFOLIO_RECOMMENDATION" and params["budget"] is None:
        print(f"[4] Route: ASK FOR DETAILS (missing budget)")
        reply = "I'd be happy to recommend a portfolio! Please provide your monthly SIP amount, investment horizon, and risk tolerance."
        print(f"\nRESPONSE:\n{reply}")
        return

    # Step 7: Fallback to LLM
    print(f"[4] Route: LLM GENERATION")
    reply = generate_chat_reply(
        model, tokenizer, clean_input, device,
        max_new_tokens=250,
        temperature=0.5,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4
    )

    is_valid, reason = validate_llm_response(reply, clean_input)
    if is_valid:
        print(f"[5] Response Validation: PASSED")
    else:
        print(f"[5] Response Validation: FAILED ({reason})")

    print(f"\nRESPONSE:\n{reply}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*80)
    print("       INTEGRATED CHAT.PY TESTING (Full Pipeline)")
    print("="*80)

    # Load tokenizer
    tokenizer = Tokenizer("tokenizer/tokenizer_v2.model")

    # Load model
    cfg_dict = load_config("configs/model_config.yaml")
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    model = MentraFiAI(cfg).to(device)

    ckpt_path = Path("checkpoints/finetune_v3/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/finetune_v2/best_model.pt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded: {ckpt_path.name}")

    # Load classifier
    classifier = None
    clf_path = Path("checkpoints/classifier/best_classifier.pt")
    if clf_path.exists():
        try:
            from model.classifier import MentraIntentClassifier, ClassifierConfig
            clf_ckpt = torch.load(clf_path, map_location=device, weights_only=False)
            clf_cfg = clf_ckpt.get("cfg", ClassifierConfig())
            classifier = MentraIntentClassifier(clf_cfg).to(device)
            classifier.load_state_dict(clf_ckpt["model_state_dict"])
            classifier.eval()
            print(f"✓ Classifier loaded")
        except Exception as e:
            print(f"⚠ Classifier not available: {e}")

    # Load database
    db = FundDatabase()
    if db.is_connected:
        print(f"✓ Database connected: {db.get_fund_count():,} funds")
    else:
        print(f"⚠ Database not connected")

    print("\n" + "="*80)

    # Test queries
    test_queries = [
        "Hello",
        "What is SIP?",
        "I am 28 years old, want to invest Rs 5000 per month, moderate risk, 10 year horizon",
        "I am 35, want aggressive growth with Rs 10000 monthly SIP for 20 years",
        "I want to accumulate Rs 1 crore in 15 years",
        "How is LTCG calculated?",
        "Suggest me best mutual fund",
    ]

    for query in test_queries:
        simulate_chat_interaction(query, model, tokenizer, device, classifier, db)

    print("\n" + "="*80)
    print("SUMMARY OF ROUTING:")
    print("="*80)
    print("✓ Greetings → Pre-defined response (NO LLM)")
    print("✓ Definitions/Tax → Glossary lookup (NO LLM)")
    print("✓ Portfolio w/ budget → Grounded generator (NO LLM for math)")
    print("✓ Portfolio w/o budget → Ask for details (NO LLM)")
    print("⚠ Generic queries → LLM generation (with validation)")
    print("\nThe key fix: Financial calculations NEVER go through free-form LLM generation!")
    print("="*80)


if __name__ == "__main__":
    main()
