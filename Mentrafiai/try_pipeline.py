"""Run real questions through the full serve.py pipeline (Networks 1+2, math engine, DB, Network 3).

    python try_pipeline.py            # built-in questions
    python try_pipeline.py "your question" "another question"
"""
import logging
import sys

import serve

QUESTIONS = [
    "I am 28, I can invest 15000 per month for 10 years, moderate risk. Suggest a portfolio.",
    "I am a conservative investor with Rs 8000 monthly for 2 years. Which funds should I choose?",
    "Calculate SIP of 10000 for 10 years at 12%",
    "How much SIP do I need to reach 1 crore in 15 years?",
    "I sold equity fund units after 18 months with a profit of 3 lakh, how much tax?",
    "What is ELSS and how does 80C work?",
    "Recommend best flexi cap funds",
    "I am 35 and want to save for my child's education in 12 years, I can do 20000 monthly, aggressive.",
    "Why do markets fall?",
    "Should I invest in bitcoin?",
]


def main() -> None:
    logging.getLogger("mentrafiai-server").setLevel(logging.INFO)
    engine = serve.engine
    engine.load()
    print("\nNetwork 3:", engine.checkpoint_name, "\n")
    questions = sys.argv[1:] or QUESTIONS
    for question in questions:
        reply, funds, intent, risk = engine.process_query(question)
        print("=" * 100)
        print("Q:", question)
        print(f"intent={intent} risk={risk} fund_cards={[f.name for f in funds]}")
        print(reply[:1800])


if __name__ == "__main__":
    main()
