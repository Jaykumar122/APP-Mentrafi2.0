"""Concept questions must map to the right curriculum topic; advice requests must not."""
import unittest

from data_prep.mutual_fund_curriculum import TOPICS
from inference.definition_router import ALIASES, faithful, is_concept_question, match_topic, reference


class DefinitionRouterTests(unittest.TestCase):
    def test_topics_exist(self):
        for _, topic in ALIASES:
            self.assertIn(topic, TOPICS)

    def test_concept_questions(self):
        cases = {
            "what is sip?": "sip_definition",
            "What is SIP": "sip_definition",
            "what is the difference between sip and lumpsum?": "lump_sum",
            "SIP vs lump sum which is better": "lump_sum",
            "difference between direct and regular plan": "direct_and_regular",
            "what is nav": "nav_formula",
            "Is a lower NAV fund cheaper?": "nav_and_price_myth",
            "explain expense ratio": "total_expense_ratio",
            "what is xirr": "xirr",
            "what is elss": "elss_lock_in",
            "what is a liquid fund": "liquid_fund",
            "difference between multi cap and flexi cap": "multi_cap_fund",
            "what is a flexi cap fund?": "flexi_cap_fund",
            "what is step up sip": "step_up_sip",
            "what is exit load": "exit_load",
            "What is a mutual fund?": "mutual_fund",
            "what happens if i stop my sip?": "sip_pause",
            "how does swp work": "swp",
            "what is kyc": "kyc",
            "what is cagr": "cagr",
        }
        for question, topic in cases.items():
            self.assertTrue(is_concept_question(question), question)
            self.assertEqual(match_topic(question), topic, question)

    def test_advice_requests_are_not_definitions(self):
        for question in ["Which mutual funds should I buy?", "Suggest a portfolio for me",
                         "I can invest 5000 monthly SIP for 15 years", "Calculate SIP of 10000 for 10 years",
                         "best flexi cap funds?", "what is the nav of hdfc flexi cap",
                         "where should I invest 20k monthly?"]:
            self.assertFalse(is_concept_question(question), question)

    def test_faithful(self):
        _, answer = reference("sip_definition")
        self.assertTrue(faithful(answer, "sip_definition"))
        self.assertFalse(faithful("Sip is a general education component for a child.", "sip_definition"))


if __name__ == "__main__":
    unittest.main()
