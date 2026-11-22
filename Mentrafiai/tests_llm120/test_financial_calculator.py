"""Regression tests for the deterministic math engine (inference/financial_calculator.py)."""

import math
import unittest

from inference import financial_calculator as fc


def simulate_sip(payment, months, annual_pct, step_up_pct=0.0, timing="start"):
    """Independent month-by-month cash-flow simulation."""
    r = annual_pct / 100 / 12
    balance = 0.0
    installment = payment
    for month in range(months):
        if month and month % 12 == 0:
            installment *= 1 + step_up_pct / 100
        if timing == "start":
            balance += installment
            balance *= 1 + r
        else:
            balance *= 1 + r
            balance += installment
    return balance


class SipTests(unittest.TestCase):
    def test_reference_value_matches_common_indian_calculators(self):
        # ₹10,000/month, 10 years, 12%: widely published result ₹23,23,391.
        res = fc.calculate_sip(10000, 10, 12)
        self.assertEqual(round(res["corpus"]), 2323391)
        self.assertEqual(res["invested"], 1200000)
        self.assertEqual(res["months"], 120)

    def test_closed_form_matches_simulation(self):
        for payment, years, rate in [(500, 1, 8), (5000, 15, 12), (25000, 25, 14.5), (1000, 0.5, 7)]:
            for timing in ("start", "end"):
                res = fc.calculate_sip(payment, years, rate, timing=timing)
                expected = simulate_sip(payment, int(round(years * 12)), rate, timing=timing)
                self.assertAlmostEqual(res["corpus"], expected, delta=1e-6 * expected)

    def test_zero_and_negative_rates_do_not_crash(self):
        self.assertEqual(fc.calculate_sip(1000, 5, 0)["corpus"], 60000)
        negative = fc.calculate_sip(1000, 5, -5)["corpus"]
        self.assertLess(negative, 60000)
        self.assertAlmostEqual(negative, simulate_sip(1000, 60, -5), places=6)

    def test_invalid_inputs_raise(self):
        for args in [(-1, 5, 12), (1000, 0, 12), (1000, 5, float("nan")), (1000, 5, -100)]:
            with self.assertRaises(ValueError):
                fc.calculate_sip(*args)

    def test_step_up_matches_simulation(self):
        res = fc.calculate_step_up_sip(5000, 10, 12, 10)
        self.assertAlmostEqual(res["corpus"], simulate_sip(5000, 120, 12, 10), places=4)
        invested = sum(5000 * 1.1 ** year * 12 for year in range(10))
        self.assertAlmostEqual(res["invested"], invested, places=4)
        self.assertGreater(res["corpus"], fc.calculate_sip(5000, 10, 12)["corpus"])

    def test_goal_sip_reaches_target_with_whole_rupee(self):
        res = fc.calculate_goal_sip(10000000, 15, 12)
        self.assertEqual(res["required_sip"], 19819)
        self.assertGreaterEqual(fc.calculate_sip(res["required_sip"], 15, 12)["corpus"], 10000000)
        self.assertLess(fc.calculate_sip(res["required_sip"] - 1, 15, 12)["corpus"], 10000000)

    def test_goal_step_up_start_is_exact(self):
        res = fc.calculate_goal_step_up_sip(10000000, 15, 12, 10)
        start = res["starting_sip"]
        self.assertGreaterEqual(simulate_sip(start, 180, 12, 10), 10000000)
        self.assertLess(simulate_sip(start - 1, 180, 12, 10), 10000000)
        self.assertLess(start, fc.calculate_goal_sip(10000000, 15, 12)["required_sip"])

    def test_lumpsum_and_inflation(self):
        res = fc.calculate_lumpsum(100000, 10, 12)
        self.assertAlmostEqual(res["corpus"], 100000 * 1.12 ** 10, places=6)
        self.assertAlmostEqual(res["real_corpus"], res["corpus"] / 1.06 ** 10, places=6)
        self.assertAlmostEqual(fc.real_return_pct(12, 6), (1.12 / 1.06 - 1) * 100)
        self.assertAlmostEqual(fc.calculate_cagr(100, 200, 6), (2 ** (1 / 6) - 1) * 100)

    def test_swp(self):
        res = fc.calculate_swp(1000000, 10000, 0)
        self.assertEqual(res["months_fully_paid"], 100)
        self.assertEqual(res["remaining_corpus"], 0)
        forever = fc.calculate_swp(1200000, 5000, 6, max_years=30)
        self.assertTrue(forever["sustainable_forever"])
        self.assertTrue(forever["lasts_full_period"])


class TaxTests(unittest.TestCase):
    def test_equity_ltcg_uses_exemption_and_cess(self):
        res = fc.calculate_capital_gains_tax(300000, 18, "equity")
        self.assertEqual(res["section"], "Section 112A")
        self.assertEqual(res["taxable_gain"], 175000)
        self.assertAlmostEqual(res["total_tax"], 175000 * 0.125 * 1.04)

    def test_equity_ltcg_exemption_is_shared_within_year(self):
        res = fc.calculate_capital_gains_tax(100000, 24, "equity", other_equity_ltcg_this_fy=100000)
        self.assertEqual(res["exemption_used"], 25000)
        self.assertEqual(res["taxable_gain"], 75000)

    def test_equity_stcg_and_boundary(self):
        self.assertEqual(fc.calculate_capital_gains_tax(100000, 12, "equity")["term"], "short-term")
        self.assertAlmostEqual(fc.calculate_capital_gains_tax(100000, 6, "equity")["total_tax"], 20800)

    def test_debt_and_other_funds(self):
        debt = fc.calculate_capital_gains_tax(100000, 60, "debt")
        self.assertTrue(debt["needs_slab_rate"])
        self.assertIsNone(debt["total_tax"])
        self.assertAlmostEqual(fc.calculate_capital_gains_tax(100000, 60, "debt", slab_rate_pct=30)["total_tax"], 31200)
        self.assertAlmostEqual(fc.calculate_capital_gains_tax(100000, 25, "other")["total_tax"], 13000)
        self.assertTrue(fc.calculate_capital_gains_tax(100000, 24, "other")["needs_slab_rate"])
        self.assertEqual(fc.calculate_capital_gains_tax(-5000, 30, "equity")["total_tax"], 0)

    def test_80c(self):
        self.assertAlmostEqual(fc.calculate_80c_saving(200000, 30)["tax_saved"], 46800)
        self.assertEqual(fc.calculate_80c_saving(100000, 30, other_80c=100000)["eligible_deduction"], 50000)
        self.assertEqual(fc.calculate_80c_saving(100000, 30, regime="new")["tax_saved"], 0)


class FormattingAndParsingTests(unittest.TestCase):
    def test_formatting(self):
        self.assertEqual(fc.format_inr_exact(1234567), "₹12,34,567")
        self.assertEqual(fc.format_inr_exact(999), "₹999")
        self.assertEqual(fc.format_inr(45000), "₹45,000")
        self.assertEqual(fc.format_inr(235000), "₹2.35 Lakh")
        self.assertEqual(fc.format_inr(9999999.6), "₹1.00 Crore")
        self.assertEqual(fc.format_inr(-150000), "-₹1.50 Lakh")

    def test_amount_parsing(self):
        cases = {
            "invest 1,00,000 once": 100000,
            "SIP of 2000 for 5 years": 2000,
            "₹10,000 per month": 10000,
            "10k monthly": 10000,
            "1.5 lakh": 150000,
            "2 crore goal": 20000000,
            "Section 112A tax on 2 lakh gain": 200000,
            "I want 50 lakh by 2035": 5000000,
            "I can do 20000 monthly": 20000,
        }
        for text, expected in cases.items():
            self.assertEqual(fc.parse_currency_amount(text), expected, text)
        self.assertIsNone(fc.parse_currency_amount("I am 30 years old with a 15 year horizon at 12%"))
        amounts = fc.extract_amounts("I can invest 10k per month, how do I reach 1 crore?")
        self.assertEqual([(a["amount"], a["monthly"]) for a in amounts], [(10000, True), (10000000, False)])


class DetectTests(unittest.TestCase):
    def test_informational_questions_are_not_hijacked(self):
        for query in ["What is LTCG?", "What is the LTCG exemption limit of 1.25 lakh?",
                      "Explain SIP", "What is section 80C limit of 1.5 lakh?"]:
            self.assertIsNone(fc.detect_and_calculate(query), query)

    def test_sip_report_numbers(self):
        report = fc.detect_and_calculate("Calculate SIP of 10000 for 10 years")
        self.assertIn("₹23.23 Lakh", report)
        self.assertIn("₹12.00 Lakh", report)

    def test_goal_report(self):
        report = fc.detect_and_calculate("How much SIP do I need to reach 1 crore in 15 years?")
        self.assertIn("₹19,819 / month", report)
        self.assertNotIn("0.72", report)

    def test_goal_with_budget(self):
        report = fc.detect_and_calculate("I can invest 10k per month, can I reach 1 crore in 15 years?")
        self.assertIn("Your budget of ₹10,000/month", report)
        self.assertIn("short of the target", report)

    def test_rate_and_step_up_are_separated(self):
        report = fc.detect_and_calculate("SIP of 5000 for 10 years with 15% step up at 12% return")
        expected = fc.calculate_step_up_sip(5000, 10, 12, 15)["corpus_fmt"]
        self.assertIn(expected, report)
        self.assertIn("at 12.0% a year", report)

    def test_lumpsum_and_tax_reports(self):
        self.assertIn(fc.calculate_lumpsum(100000, 5, 12)["corpus_fmt"],
                      fc.detect_and_calculate("lumpsum of 1 lakh for 5 years"))
        tax = fc.detect_and_calculate("I made a profit of 3 lakh on equity fund held for 18 months, how much tax?")
        self.assertIn("₹22,750", tax)
        saving = fc.detect_and_calculate("How much tax can I save with 1.5 lakh in ELSS at 30% slab?")
        self.assertIn("₹46,800", saving)

    def test_horizon_from_ages(self):
        report = fc.detect_and_calculate("I am 30, what corpus will 10000 per month build till 60?")
        self.assertIn("30 years", report)

    def test_portfolio_requests_go_to_the_advisor(self):
        for query in ["I am 30, I can invest 10000 per month for 10 years, suggest a portfolio",
                      "Recommend best funds for a 5000 SIP for 15 years",
                      "Where should I invest 20k monthly for 7 years?"]:
            self.assertIsNone(fc.detect_and_calculate(query), query)
        self.assertIsNotNone(fc.detect_and_calculate("Suggest funds and calculate the corpus for 5000 SIP for 10 years"))


if __name__ == "__main__":
    unittest.main()
