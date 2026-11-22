"""Regression checks for authored education, not independent financial certification."""
import json
import tempfile
import unittest
from pathlib import Path

from data_prep.audit_llm120 import binary_documents
from data_prep.build_llm120_sft import (
    HINGLISH, calculation_records, curriculum_training_split, english_question, planning_answer, sip_value,
)
from data_prep.mutual_fund_curriculum import TOPICS, curriculum_records
from llm120.data import _validated_sft
from llm120.prepare import prepare_dataset
from llm120.tokenization import Tokenizer, validate_messages

ROOT = Path(__file__).resolve().parents[1]


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer(ROOT / "tokenizer/tokenizer_v2.model")

    def test_definitions_conversation_order_and_lengths(self):
        records = curriculum_records()
        self.assertGreaterEqual(len(TOPICS), 90)
        self.assertEqual(set(TOPICS), {r["topic"] for r in records if r["topic"] != "english_communication"})
        self.assertGreaterEqual(sum(r["topic"] == "english_communication" for r in records), 15)
        for record in records + calculation_records():
            with self.subTest(topic=record["topic"]):
                validate_messages(record["messages"])
                ids, _ = _validated_sft(record, self.tok, 1024)
                self.assertLessEqual(len(ids), 1025)
                for message in record["messages"][1:]:
                    self.assertFalse(HINGLISH.search(message["content"]))
                    self.assertFalse(any(c.isalpha() and not c.isascii() for c in message["content"]))

    def test_language_gate(self):
        self.assertTrue(english_question("Recommend funds for a monthly SIP."))
        self.assertFalse(english_question("Recommend karo 5 saal ke liye"))
        self.assertFalse(english_question("मुझे SIP बताओ"))

    def test_plans_do_not_teach_short_term_equity_or_invent_names(self):
        for risk in ("conservative", "moderate", "aggressive"):
            answer = planning_answer(5000, 3, risk)
            self.assertIn("equity-heavy portfolio", answer)
            self.assertIn("can lose", answer)
            self.assertIn("No verified scheme facts", answer)
            self.assertNotIn("SBI", answer)
            self.assertNotIn("Hello", answer)
        with self.assertRaises(ValueError):
            planning_answer(0, 3, "moderate")

    def test_annuity_matches_independent_monthly_cashflow_simulation(self):
        for payment in (500, 2000):
            for years in (1, 5, 15):
                for annual in (0, 0.08):
                    for beginning in (False, True):
                        balance = 0.0
                        for _ in range(years * 12):
                            if beginning:
                                balance += payment
                            balance *= 1 + annual / 12
                            if not beginning:
                                balance += payment
                        self.assertAlmostEqual(sip_value(payment, years, annual, beginning=beginning), balance, places=6)

    def test_curriculum_split_preserves_all_definitions_and_disjoint_questions(self):
        records = curriculum_records() + calculation_records()
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.jsonl", Path(tmp) / "dataset"
            source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            manifest = prepare_dataset([source], output, self.tok, mode="sft", context_length=1024, val_fraction=0.1)
            curriculum_training_split(output, records, self.tok, manifest)
            questions = {}
            for split in ("train", "val"):
                examples = [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                questions[split] = {r["messages"][1]["content"] for r in examples}
                self.assertEqual(manifest["splits"][split]["documents"], len(examples))
            self.assertTrue({r["question"] for r in TOPICS.values()} <= questions["train"])
            self.assertFalse(questions["train"] & questions["val"])
            self.assertTrue(questions["val"])

    def test_incomplete_binary_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.bin"
            path.write_bytes(b"\x04\x00\x05\x00")
            with self.assertRaisesRegex(ValueError, "no EOS"):
                list(binary_documents(path, 2))


if __name__ == "__main__":
    unittest.main()
