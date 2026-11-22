"""Grounded Network 3 format: training examples and serving prompts must agree."""

import unittest

from data_prep.build_grounded_sft import GENERATORS, grounded_records
from llm120.grounding import (Allocation, FundFact, GroundedContext, build_messages, fund_fact_from_row,
                              to_display_text, to_model_text, ungrounded_numbers)
from llm120.tokenization import Tokenizer, serialize
from pathlib import Path

TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer_v2.model"


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer(TOKENIZER)

    def test_render_contains_every_fact_and_uses_rs(self):
        ctx = GroundedContext(
            intent="PORTFOLIO_RECOMMENDATION", risk="moderate",
            profile={"name": "Asha", "monthly_budget": "₹10,000", "horizon": "10 years"},
            calculations={"total invested": "₹12.00 Lakh", "projected corpus": "₹23.23 Lakh"},
            allocation=[Allocation("Flexi Cap Fund", "₹10,000", "100%", 1)],
            funds=[FundFact("Kaveri Flexi Cap Fund - Direct Growth", "Flexi Cap Fund", nav="₹123.45",
                            return_3y="15.1%")],
            data_as_of="10-May-2026")
        text = ctx.render()
        for fact in ("Asha", "Rs 10,000", "Rs 23.23 Lakh", "Kaveri Flexi Cap Fund - Direct Growth", "NAV Rs 123.45",
                     "3y 15.1%", "10-May-2026", "fund [1]"):
            self.assertIn(fact, text)
        self.assertNotIn("₹", text)
        messages = build_messages("Suggest funds for ₹10,000", ctx)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "Suggest funds for Rs 10,000")
        ids, _ = serialize(messages, self.tok, add_generation_prompt=True)
        self.assertLess(len(ids), 600)

    def test_number_guard(self):
        source = "projected corpus Rs 23.23 Lakh; total invested Rs 12,00,000; 3y 15.1%"
        self.assertEqual(ungrounded_numbers("You invest Rs 12,00,000 and may get Rs 23.23 Lakh.", source), set())
        self.assertEqual(ungrounded_numbers("Expect 18% returns and Rs 30 Lakh.", source), {"18", "30"})
        self.assertEqual(ungrounded_numbers("Step 1 of 3.", source), set())

    def test_display_round_trip(self):
        self.assertEqual(to_display_text(to_model_text("₹1,234 and ₹ 5 Lakh")), "₹1,234 and ₹5 Lakh")
        self.assertEqual(to_display_text("Rs. 500"), "₹500")

    def test_fund_row_conversion(self):
        fact = fund_fact_from_row({"name": "X Fund (Formerly Y Fund) - Direct Plan - Growth", "subcategory": "Large Cap",
                                   "nav": 101.5, "return_1y": 12.345, "return_3y": None, "expense_ratio": 0.5,
                                   "sebi_riskometer": "Very High", "plan_type": "Direct", "option_type": "Growth"})
        self.assertEqual(fact.name, "X Fund - Direct Plan - Growth")
        self.assertEqual(fact.nav, "₹101.50")
        self.assertEqual(fact.return_1y, "12.3%")
        self.assertIsNone(fact.return_3y)
        self.assertNotIn("0.5", fact.render(1))
        self.assertNotIn("Very High", fact.render(1))

    def test_generated_examples_are_grounded_and_fit(self):
        records = grounded_records(400, seed=1)
        self.assertEqual({r["topic"] for r in records}, {name for name, _, _ in GENERATORS})
        for record in records:
            messages = record["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[-1]["role"], "assistant")
            answer = messages[-1]["content"]
            self.assertNotIn("₹", answer)
            self.assertEqual(ungrounded_numbers(answer, *(m["content"] for m in messages[:-1])), set(), answer)
            if record["topic"] == "grounded_portfolio":
                names = [line.split(":")[0].split("] ", 1)[1] for line in messages[0]["content"].split("[")[1:]
                         if "] " in line and ":" in line]
                self.assertTrue(any(name in answer for name in names) or "no verified scheme" in answer)

    def test_reply_problems(self):
        import serve

        context = "Investor: name Asha; monthly SIP budget Rs 8,000. projected corpus Rs 2.09 Lakh. NAV Rs 10.00"
        good = "Thank you, Asha. You invest Rs 8,000 a month and reach Rs 2.09 Lakh."
        self.assertEqual(serve.reply_problems(good, context, ["Rs 8,000", "Rs 2.09 Lakh"]), [])
        self.assertTrue(serve.reply_problems("Thank you, Suresh. Rs 8,000 and Rs 2.09 Lakh.", context))
        self.assertTrue(serve.reply_problems("Its expense ratio is 0.5%. Rs 8,000 Rs 2.09 Lakh.", context))
        self.assertTrue(serve.reply_problems("It has a Low riskometer. Rs 8,000 Rs 2.09 Lakh.", context))
        advice = "Also compare costs, portfolio overlap and fund manager tenure in the latest factsheets."
        self.assertEqual(serve.reply_problems(advice, context), [])
        self.assertTrue(serve.reply_problems("You invest Rs 800 a month for Rs 2.09 Lakh.", context, ["Rs 8,000"]))

    def test_polish_reply(self):
        import serve

        reply = ("For a aggressive investor. You did not state your risk appetite, so the plan uses the "
                 "aggressive profile; tell me if that is wrong. Put Rs 100 in it.")
        fixed = serve.polish_reply(reply, "I am an aggressive investor")
        self.assertEqual(fixed, "For an aggressive investor. Put Rs 100 in it.")
        self.assertIn("did not state", serve.polish_reply(reply, "Suggest funds for 5000 a month"))

    def test_generation_is_deterministic(self):
        self.assertEqual(grounded_records(30, seed=5), grounded_records(30, seed=5))


if __name__ == "__main__":
    unittest.main()
