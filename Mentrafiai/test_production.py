"""
Test the ACTUAL production routing (serve.py's MentraFiAIEngine.process_query),
not the raw generate_chat_reply() bypass that test_llm.py used.

This tells us what real users actually experience, since serve.py only calls
the raw 102.5M generative network for:
  1. Pure greetings (rare - templates cover most cases)
  2. The final catch-all fallback for general/open-ended questions
Everything else (portfolio numbers, goal SIP math, category search, glossary)
is fully grounded/deterministic and never touches the generative network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from serve import MentraFiAIEngine

QUERIES = [
    # Should hit grounded calculator / portfolio paths (no LLM)
    "I am 28 years old, want to invest Rs 5000 per month, moderate risk, 10 year horizon",
    "I want to accumulate Rs 1 crore in 15 years. What should I do?",
    "recommend best large cap funds",

    # Should hit glossary (no LLM)
    "What is SIP?",
    "How is LTCG calculated on equity funds?",

    # Should hit the LLM FALLBACK (real test of Network 3 quality)
    "Suggest me best mutual fund",
    "Is it safe to invest during a market crash?",
    "What happens to my SIP if the fund house shuts down?",
    "Should I stop my SIP when the market is falling?",
    "How do I choose between two similar mutual funds?",
    "What is the difference between a mutual fund and a stock?",
]


def main():
    engine = MentraFiAIEngine()
    print("Loading production engine (this mirrors serve.py exactly)...")
    engine.load()
    print("\n" + "=" * 80)

    for q in QUERIES:
        reply, funds, intent, risk = engine.process_query(q)
        print(f"\nQUERY: {q}")
        print(f"INTENT: {intent}")
        print(f"REPLY:\n{reply}")
        print("-" * 80)


if __name__ == "__main__":
    main()
