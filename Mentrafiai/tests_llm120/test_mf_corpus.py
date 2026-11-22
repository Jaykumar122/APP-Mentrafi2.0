"""Indian-MF pretraining corpus builders: classification, filtering, generated numbers."""

import random
import re
import unittest
from datetime import date

from data_prep import build_mf_pretrain as build
from data_prep import gen_mf_pretrain as gen
from inference import financial_calculator as fc


class CorpusTests(unittest.TestCase):
    def test_category_classification(self):
        cases = {
            "Motilal Oswal Large and Midcap Fund - Direct Plan - Growth": "Large and Mid Cap Fund",
            "HDFC Mid Cap Fund - Direct Plan - Growth Option": "Mid Cap Fund",
            "Parag Parikh Flexi Cap Fund - Direct Plan - Growth": "Flexi Cap Fund",
            "SBI Liquid Fund - Direct Plan - Growth": "Liquid Fund",
            "Kotak FMP Series 172 - Direct": "Fixed Maturity Plan",
            "Axis ELSS Tax Saver Fund - Direct Growth": "ELSS Tax Saver Fund",
            "ICICI Prudential Equity Arbitrage Fund - Direct Plan - Growth": "Arbitrage Fund",
            "UTI Nifty 50 Index Fund - Direct Plan - Growth": "Index Fund",
        }
        for name, expected in cases.items():
            self.assertEqual(gen.classify(name)[0], expected, name)
        self.assertEqual(gen.base_scheme("UTI Flexi Cap Fund - Direct Plan - Growth"), "UTI Flexi Cap Fund")

    def test_fund_doc_uses_only_verified_fields(self):
        row = {"name": "Example Flexi Cap Fund - Direct Plan - Growth", "fund_house": "Example Mutual Fund",
               "plan_type": "Direct", "option_type": "Growth", "nav": 123.4567, "nav_date": date(2026, 9, 25),
               "launch_date": date(2013, 1, 2), "isin": "INF000000000", "return_1y": 12.3, "return_3y": None,
               "return_5y": 15.0, "return_10y": None, "expense_ratio": 9.99, "aum_crores": 99999}
        text = gen.fund_doc(random.Random(0), row)
        self.assertIn("Rs 123.4567", text)
        self.assertIn("25 September 2026", text)
        self.assertIn("12.30% over 1 year", text)
        self.assertNotIn("9.99", text)
        self.assertNotIn("99,999", text)

    def test_calculator_documents_match_engine(self):
        rng = random.Random(7)
        for _ in range(300):
            text = gen.calc_doc(rng)
            self.assertNotIn("₹", text)
            match = re.search(r"invests Rs ([\d,]+) at the start of every month for (\d+) years\. Assuming (\d+)%", text)
            if match:
                sip, years, rate = (int(match.group(i).replace(",", "")) for i in (1, 2, 3))
                corpus = fc.format_inr(fc.calculate_sip(sip, years, rate)["corpus"]).replace("₹", "Rs ")
                self.assertIn(f"projected value is {corpus}", text)

    def test_brand_filter_and_split(self):
        self.assertTrue(build.REAL_BRAND.search("Invest in HDFC Flexi Cap Fund today"))
        self.assertTrue(build.REAL_BRAND.search("quant active fund returned 30%"))
        self.assertIsNone(build.REAL_BRAND.search("A flexi cap fund invests across market caps"))
        self.assertEqual(build.is_val("abc", 0.01), build.is_val("abc", 0.01))
        share = sum(build.is_val(str(i), 0.01) for i in range(20000)) / 20000
        self.assertAlmostEqual(share, 0.01, delta=0.004)


if __name__ == "__main__":
    unittest.main()
