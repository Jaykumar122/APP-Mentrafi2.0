"""
Corpus Quality & Cleanliness Audit Script.
Scans raw and generated files for garbage, repetition, mojibake, HTML tags, and anomalous character distributions.
"""

import json
import re
from pathlib import Path
from collections import Counter

FILES_TO_AUDIT = [
    Path("data/raw/corpus_v2/advanced_finance_math.jsonl"),
    Path("data/raw/corpus_v2/conversational_dialogues.jsonl"),
    Path("data/raw/corpus_v2/indian_market_knowledge.jsonl"),
    Path("data/raw/corpus_v2/four_pillars_mastery.jsonl"),
    Path("data/raw/sft/conversational_greetings.jsonl"),
]

MOJIBAKE_PATTERNS = re.compile(r"(â€™|â€˜|â€œ|â€\x9d|â€“|â€”|â€¦|Â|Ã©|Ã¨|Ã¡|Ã­|Ã³|Ãº|Ã±|\ufffd)")
HTML_TAGS = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(\s+[^>]*)?>")


def audit_jsonl(path: Path, max_samples: int = 10000):
    if not path.exists():
        print(f"[SKIP] {path} does not exist.")
        return

    print(f"\nAuditing: {path} (sampling up to {max_samples:,} records)...")
    total = 0
    empty_lines = 0
    mojibake_hits = 0
    html_hits = 0
    repetition_hits = 0
    anomalous_char_hits = 0
    word_lengths = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if not line.strip():
                empty_lines += 1
                continue
            
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("assistant") or ""
            except Exception:
                continue

            if not text or len(text.strip()) < 10:
                empty_lines += 1
                continue

            # Check mojibake
            if MOJIBAKE_PATTERNS.search(text):
                mojibake_hits += 1

            # Check HTML
            if HTML_TAGS.search(text):
                html_hits += 1

            # Check word length distribution
            words = text.split()
            if words:
                avg_wlen = sum(len(w) for w in words) / len(words)
                word_lengths.append(avg_wlen)

                # Check repetition
                wc = Counter(words)
                most_common_w, most_common_cnt = wc.most_common(1)[0]
                if most_common_cnt > len(words) * 0.25 and len(most_common_w) > 3:
                    repetition_hits += 1

            # Check character anomalies (too many non-alphanumeric)
            alpha_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / len(text)
            if alpha_ratio < 0.70:
                anomalous_char_hits += 1

            if total >= max_samples:
                break

    mean_avg_wlen = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0
    print(f"  • Total samples audited: {total:,}")
    print(f"  • Mojibake / encoding errors : {mojibake_hits} ({mojibake_hits/total*100:.2f}%)")
    print(f"  • Stray HTML / code tags    : {html_hits} ({html_hits/total*100:.2f}%)")
    print(f"  • Degenerate repetitions    : {repetition_hits} ({repetition_hits/total*100:.2f}%)")
    print(f"  • Symbol/binary anomalies   : {anomalous_char_hits} ({anomalous_char_hits/total*100:.2f}%)")
    print(f"  • Mean word length          : {mean_avg_wlen:.2f} chars (Healthy range: 4.0 - 6.5)")
    
    clean_score = 100.0 - ((mojibake_hits + html_hits + repetition_hits + anomalous_char_hits) / total * 100.0)
    print(f"  -> Quality / Cleanliness Score: {clean_score:.2f}%")


def main():
    print("=" * 60)
    print("MentraFiAI Corpus Cleanliness & Quality Audit")
    print("=" * 60)
    for p in FILES_TO_AUDIT:
        audit_jsonl(p)


if __name__ == "__main__":
    main()
