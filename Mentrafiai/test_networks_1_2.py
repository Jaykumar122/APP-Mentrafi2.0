"""
Standalone test for Network 1 (10.65M Intent Classifier) and
Network 2 (15.26M Query Parser / entity extractor), isolated from
Network 3 (the generative LLM) and from the DB.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.query_parser import FinancialQueryParser

TEST_CASES = [
    # (query, expected_intent_loose, expected_age, expected_amount, expected_horizon, expected_risk)
    ("I am 28 years old, want to invest Rs 5000 per month, moderate risk, 10 year horizon",
     "PORTFOLIO_RECOMMENDATION", 28, 5000, 10, "moderate"),
    ("I want to accumulate Rs 1 crore in 15 years. What should I do?",
     None, None, 10000000, 15, None),
    ("What is SIP?", "DEFINITION", None, None, None, None),
    ("How is LTCG calculated on equity funds?", "DEFINITION", None, None, None, None),
    ("recommend best large cap funds", None, None, None, None, None),
    ("I am 35, want aggressive growth with Rs 10000 monthly SIP for 20 years",
     "PORTFOLIO_RECOMMENDATION", 35, 10000, 20, "aggressive"),
    ("Compare SBI Small Cap vs Nippon India Small Cap", "FUND_COMPARISON", None, None, None, None),
    ("Hello", "GREETING", None, None, None, None),
    ("40 year old, 2.5 lakh lumpsum, conservative risk, 5 years",
     None, 40, 250000, 5, "conservative"),
    ("Suggest me best mutual fund", None, None, None, None, None),
]


def main():
    device = "cuda"
    tok_path = "tokenizer/tokenizer_v2.model"
    clf_path = Path("checkpoints/classifier/best_classifier.pt")
    parser_path = Path("checkpoints/query_parser/best_parser_net.pt")

    parser = FinancialQueryParser(
        classifier_ckpt=str(clf_path) if clf_path.exists() else None,
        parser_ckpt=str(parser_path) if parser_path.exists() else None,
        tokenizer_path=tok_path,
        device=device,
    )

    print(f"Classifier loaded: {parser.classifier is not None}")
    print(f"Parser net loaded: {parser.parser_net is not None}")
    print("=" * 100)

    correct_intent = 0
    correct_age = 0
    correct_amount = 0
    correct_horizon = 0
    correct_risk = 0
    total_intent_checked = 0
    total_age_checked = 0
    total_amount_checked = 0
    total_horizon_checked = 0
    total_risk_checked = 0

    for query, exp_intent, exp_age, exp_amount, exp_horizon, exp_risk in TEST_CASES:
        parsed = parser.parse(query)
        print(f"\nQUERY: {query}")
        print(f"  -> intent={parsed.intent} (conf={parsed.intent_confidence:.2%}) "
              f"age={parsed.age} amount={parsed.amount} horizon={parsed.horizon} risk={parsed.risk}")

        if exp_intent is not None:
            total_intent_checked += 1
            ok = parsed.intent == exp_intent
            correct_intent += ok
            print(f"  [intent] expected={exp_intent} got={parsed.intent} {'OK' if ok else 'MISMATCH'}")
        if exp_age is not None:
            total_age_checked += 1
            ok = parsed.age == exp_age
            correct_age += ok
            print(f"  [age] expected={exp_age} got={parsed.age} {'OK' if ok else 'MISMATCH'}")
        if exp_amount is not None:
            total_amount_checked += 1
            ok = parsed.amount == exp_amount
            correct_amount += ok
            print(f"  [amount] expected={exp_amount} got={parsed.amount} {'OK' if ok else 'MISMATCH'}")
        if exp_horizon is not None:
            total_horizon_checked += 1
            ok = parsed.horizon == exp_horizon
            correct_horizon += ok
            print(f"  [horizon] expected={exp_horizon} got={parsed.horizon} {'OK' if ok else 'MISMATCH'}")
        if exp_risk is not None:
            total_risk_checked += 1
            ok = parsed.risk == exp_risk
            correct_risk += ok
            print(f"  [risk] expected={exp_risk} got={parsed.risk} {'OK' if ok else 'MISMATCH'}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    if total_intent_checked:
        print(f"Intent accuracy: {correct_intent}/{total_intent_checked} ({correct_intent/total_intent_checked:.0%})")
    if total_age_checked:
        print(f"Age extraction accuracy: {correct_age}/{total_age_checked} ({correct_age/total_age_checked:.0%})")
    if total_amount_checked:
        print(f"Amount extraction accuracy: {correct_amount}/{total_amount_checked} ({correct_amount/total_amount_checked:.0%})")
    if total_horizon_checked:
        print(f"Horizon extraction accuracy: {correct_horizon}/{total_horizon_checked} ({correct_horizon/total_horizon_checked:.0%})")
    if total_risk_checked:
        print(f"Risk extraction accuracy: {correct_risk}/{total_risk_checked} ({correct_risk/total_risk_checked:.0%})")


if __name__ == "__main__":
    main()
