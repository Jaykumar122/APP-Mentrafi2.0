"""
Clean, filter and deduplicate the v2 corpus: raw JSONL in -> clean JSONL out.

Pipeline per document:
    normalize -> boilerplate line strip -> quality filters -> dedup -> keep

The filter set follows the Gopher / RefinedWeb heuristics, adapted for a finance
corpus (the digit-fraction ceiling is deliberately loose because genuine
financial prose is number-dense, and a Gopher-strict threshold would throw away
exactly the content this model needs).

Every rejection is counted by reason and reported, so the cost of each filter is
visible rather than assumed.

Deduplication:
    exact    64-bit blake2b of the normalized text, GLOBAL across all sources.
    near     64-bit SimHash over word 5-grams, 4 bands x 16 bits, Hamming <= 3,
             applied PER SOURCE (structures are cleared between files to bound
             memory). Cross-source near-duplicates are therefore only caught when
             they are exact; that is a deliberate memory trade-off, not an
             oversight, and the two largest sources (fineweb-edu-dedup,
             cosmopedia-v2) are already deduplicated upstream.

Usage:
    python data_prep/clean_corpus_v2.py
    python data_prep/clean_corpus_v2.py --only finance_instruct --workers 4
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IN_DIR = Path("data/raw/corpus_v2")
OUT_DIR = Path("data/processed/corpus_v2/clean")
MIB = 1024 ** 2

# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_MOJIBAKE = {
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-",
    "â€¦": "...", "Â ": " ", "Ã©": "e", "Ã¨": "e", "Ã¡": "a", "Ã­": "i",
    "Ã³": "o", "Ãº": "u", "Ã±": "n", "â€": '"', "Â": "",
}
_SMART = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...", " ": " ", "​": "", "‌": "", "‍": "",
    "﻿": "", " ": "\n", " ": "\n",
}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINL = re.compile(r"\n{3,}")

_BOILER = re.compile(
    r"(all rights reserved|cookie polic|privacy polic|terms of (use|service)|"
    r"click here|read more|share this|follow us|sign up for|subscribe to our|"
    r"skip to (main )?content|javascript is (disabled|required)|"
    r"enable javascript|advertisement|sponsored content|"
    r"^\s*(home|menu|search|login|register|contact us|about us)\s*$|"
    r"copyright\s*(©|\(c\))|^\s*©|posted (on|by)\s|tags?:\s*$)", re.I)


def normalize(text: str) -> str:
    for k, v in _MOJIBAKE.items():
        if k in text:
            text = text.replace(k, v)
    for k, v in _SMART.items():
        if k in text:
            text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub("", text)
    lines = []
    for ln in text.split("\n"):
        ln = _MULTISPACE.sub(" ", ln).strip()
        if not ln:
            lines.append("")
            continue
        if _BOILER.search(ln) and len(ln.split()) < 25:
            continue                      # nav/legal chrome, not prose
        lines.append(ln)
    return _MULTINL.sub("\n\n", "\n".join(lines)).strip()


# --------------------------------------------------------------------------- #
# Quality filters
# --------------------------------------------------------------------------- #
STOPWORDS = {"the", "be", "is", "are", "was", "to", "of", "and", "a", "in",
             "that", "have", "has", "it", "for", "not", "on", "with", "as",
             "this", "but", "they", "at", "an", "or", "from", "by"}
_WORD = re.compile(r"[A-Za-z']+")
_SYMBOLS = re.compile(r"[#…]|\.\.\.")

MIN_WORDS = 50
MAX_WORDS = 100_000


def quality(text: str):
    """Return None if the document passes, else a short rejection reason."""
    words = text.split()
    nw = len(words)
    if nw < MIN_WORDS:
        return "too_short"
    if nw > MAX_WORDS:
        return "too_long"

    alpha_words = _WORD.findall(text)
    if len(alpha_words) < nw * 0.5:
        return "not_wordlike"
    mean_wl = sum(len(w) for w in alpha_words) / max(len(alpha_words), 1)
    if not (3.0 <= mean_wl <= 10.0):
        return "mean_word_len"

    n_alpha = sum(c.isalpha() for c in text)
    if n_alpha / len(text) < 0.55:
        return "low_alpha"
    # Loose on purpose: finance prose is number-dense.
    if sum(c.isdigit() for c in text) / len(text) > 0.30:
        return "digit_heavy"
    n_latin = sum(1 for c in text if c.isalpha() and ord(c) < 0x250)
    if n_alpha and n_latin / n_alpha < 0.90:
        return "non_latin"

    if len(_SYMBOLS.findall(text)) / nw > 0.10:
        return "symbol_heavy"

    lower = {w.lower().strip(".,;:!?\"'()") for w in words[:400]}
    if len(lower & STOPWORDS) < 2:
        return "no_stopwords"

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        if sum(1 for ln in lines if ln.rstrip().endswith("...")) / len(lines) > 0.30:
            return "ellipsis_lines"
        c = Counter(lines)
        dup_lines = sum(v - 1 for v in c.values() if v > 1)
        if dup_lines / len(lines) > 0.30:
            return "dup_lines"
        dup_chars = sum(len(k) * (v - 1) for k, v in c.items() if v > 1)
        if dup_chars / max(len(text), 1) > 0.30:
            return "dup_line_chars"

    # Repetition: how much of the document is inside a repeated 5-gram.
    if nw >= 200:
        grams = Counter(tuple(words[i:i + 5]) for i in range(nw - 4))
        rep = sum(v - 1 for v in grams.values() if v > 1)
        if rep / max(nw - 4, 1) > 0.20:
            return "ngram_repeat"
    return None


# --------------------------------------------------------------------------- #
# Hashing / SimHash
# --------------------------------------------------------------------------- #
def exact_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(text: str, k: int = 5) -> int:
    """64-bit SimHash over word k-grams (order-sensitive, cheap).

    The bit accumulation runs in numpy rather than a Python loop. The naive form
    costs 64 interpreter iterations per k-gram, and this corpus has roughly 930M
    words -- about 60 billion iterations, which dominated the whole cleaning
    pass. unpackbits does the same work in one vectorised call per document.

    Bit convention is identical to the scalar version: output bit b is the
    majority vote of hash bit b, with ties resolved to 0 (2*ones > n). numpy's
    unpackbits reads most-significant-bit first, hence the 63-b index.
    """
    words = text.lower().split()
    if len(words) < k:
        words = words + ["\x00"] * (k - len(words))
    n = len(words) - k + 1
    if n <= 0:
        return 0
    blake = hashlib.blake2b
    digests = b"".join(
        blake(" ".join(words[i:i + k]).encode("utf-8"), digest_size=8).digest()
        for i in range(n))
    cols = np.unpackbits(
        np.frombuffer(digests, dtype=np.uint8)).reshape(n, 64).sum(axis=0)
    out = 0
    for b in range(64):
        if int(cols[b]) * 2 > n:
            out |= 1 << (63 - b)
    return out


class NearDup:
    """Banded SimHash index. Cleared between sources to bound memory."""

    BANDS = 4
    BITS = 16
    MAX_BUCKET = 12
    MAX_HAMMING = 3

    def __init__(self):
        self.bands = [defaultdict(list) for _ in range(self.BANDS)]

    def is_dup(self, h: int) -> bool:
        keys = [(h >> (b * self.BITS)) & 0xFFFF for b in range(self.BANDS)]
        for b, key in enumerate(keys):
            for other in self.bands[b][key]:
                if bin(other ^ h).count("1") <= self.MAX_HAMMING:
                    return True
        for b, key in enumerate(keys):
            bucket = self.bands[b][key]
            bucket.append(h)
            if len(bucket) > self.MAX_BUCKET:
                del bucket[0]
        return False

    def clear(self):
        self.bands = [defaultdict(list) for _ in range(self.BANDS)]


# --------------------------------------------------------------------------- #
# Per-source dedup policy
#
# Near-dup detection is the right default: it is what stops a source padding its
# byte budget with rewordings of the same document. But it is a heuristic about
# INTENT -- it assumes textual similarity means redundant content -- and for one
# source that assumption is wrong by construction.
#
# finance_numeracy is generated from 8 prose templates with every number
# computed fresh. Passages from one template are near-identical in surface form
# and completely distinct in the arithmetic they demonstrate, which is the part
# the model needs to learn. Measured on the 30,139-passage set, the near-dup
# pass drops 21.0% of it -- not the whole set, because MAX_BUCKET=12 evicts old
# hashes and makes this a sliding window rather than a global index. So the
# filter is not catastrophic here, but the 21% it removes is selected for
# surface similarity while carrying distinct arithmetic, which is precisely the
# signal this source exists to provide.
#
# Exempting it is only defensible because the redundancy is bounded and declared
# rather than accidental: the generator is capped at 48 MiB (~1% of the token
# budget, ~4,000 instances per template), so the worst case if this judgement is
# wrong is 1% of the corpus being repetitive -- not a source silently inflating
# itself. Exact dedup still applies globally, which matters: the first run of
# the generator produced 16.9% byte-identical passages because three of the
# eight parameter grids were smaller than the number of draws taken from them.
# The grids were widened in response, but exact dedup stays on as the backstop.
# --------------------------------------------------------------------------- #
NO_NEAR_DUP = {"finance_numeracy"}


# --------------------------------------------------------------------------- #
# Worker (multiprocessing): the regex-heavy part
# --------------------------------------------------------------------------- #
def process_batch(lines):
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            out.append((None, "bad_json", 0, 0, None)); continue
        text = normalize(rec.get("text", ""))
        if not text:
            out.append((None, "empty_after_norm", 0, 0, None)); continue
        reason = quality(text)
        if reason:
            out.append((None, reason, 0, 0, None)); continue
        # The domain travels with the document. Dropping it here would defeat the
        # point of tagging it at fetch time: the mixer sets per-domain token
        # budgets, and reconstructing domain from the filename afterwards fails
        # for the sources that emit two domains from one stream (wikipedia
        # routes encyclopedic vs mf_domain into separate files, but a future
        # single-file router would not).
        out.append((text, None, exact_hash(text), simhash(text),
                    rec.get("domain")))
    return out


def batched(fh, size):
    buf = []
    for line in fh:
        buf.append(line)
        if len(buf) >= size:
            yield buf; buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--stats", default="data/processed/corpus_v2/clean_stats.json")
    args = ap.parse_args()

    import multiprocessing as mp
    workers = args.workers or max(1, mp.cpu_count() - 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.jsonl"))
    only = {s for s in args.only.split(",") if s}
    if only:
        files = [f for f in files if f.stem in only]
    if not files:
        print(f"no input under {IN_DIR} — run fetch_corpus_v2.py first"); return

    seen_exact = set()          # GLOBAL: cross-source exact dedup
    near = NearDup()            # per-source
    stats = {}

    print(f"cleaning {len(files)} sources with {workers} workers\n")
    with mp.Pool(workers) as pool:
        for f in files:
            t0 = time.time()
            near.clear()
            reasons = Counter()
            kept = kept_bytes = total = 0
            dup_exact = dup_near = 0
            out_path = OUT_DIR / f.name
            domain = None
            use_near = f.stem not in NO_NEAR_DUP
            if not use_near:
                print(f"  {f.stem:22} near-dup DISABLED (templated source)",
                      flush=True)
            with open(f, encoding="utf-8") as fh, \
                 open(out_path, "w", encoding="utf-8") as out:
                # domain comes from the first record; every record in a file shares it
                for res in pool.imap(process_batch, batched(fh, args.batch), chunksize=1):
                    for text, reason, eh, sh, dom in res:
                        total += 1
                        if reason:
                            reasons[reason] += 1; continue
                        if eh in seen_exact:
                            dup_exact += 1; continue
                        if use_near and near.is_dup(sh):
                            dup_near += 1; continue
                        seen_exact.add(eh)
                        if domain is None:
                            domain = dom or "?"
                        out.write(json.dumps({"text": text, "source": f.stem,
                                              "domain": dom or domain},
                                             ensure_ascii=False) + "\n")
                        kept += 1
                        kept_bytes += len(text.encode("utf-8"))
                    if total % 100_000 < args.batch:
                        print(f"  {f.stem:22} {total:>9,} read  {kept:>9,} kept  "
                              f"{kept_bytes/MIB:8.1f} MiB", flush=True)

            el = time.time() - t0
            stats[f.stem] = {
                "domain": domain or "?",
                "near_dup_applied": use_near,
                "read": total, "kept": kept, "kept_bytes": kept_bytes,
                "dup_exact": dup_exact, "dup_near": dup_near,
                "rejected": dict(reasons.most_common()),
                "keep_rate": round(kept / max(total, 1), 4),
                "seconds": round(el, 1),
            }
            print(f"  {f.stem:22} DONE  read {total:,} | kept {kept:,} "
                  f"({kept/max(total,1):.1%}) | {kept_bytes/MIB:,.1f} MiB | "
                  f"dup_exact {dup_exact:,} dup_near {dup_near:,} | {el:,.0f}s")
            top = ", ".join(f"{k}={v:,}" for k, v in reasons.most_common(5))
            print(f"  {'':22}       top rejects: {top or 'none'}\n", flush=True)

    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2))

    print("=" * 92)
    print(f'{"source":22} {"read":>11} {"kept":>11} {"keep%":>7} {"MiB":>10} {"exact":>9} {"near":>9}')
    print("-" * 92)
    tk = tb = tr = 0
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["kept_bytes"]):
        tk += s["kept"]; tb += s["kept_bytes"]; tr += s["read"]
        print(f'{name:22} {s["read"]:11,} {s["kept"]:11,} {s["keep_rate"]*100:6.1f}% '
              f'{s["kept_bytes"]/MIB:10,.1f} {s["dup_exact"]:9,} {s["dup_near"]:9,}')
    print("-" * 92)
    print(f'{"TOTAL":22} {tr:11,} {tk:11,} {tk/max(tr,1)*100:6.1f}% {tb/MIB:10,.1f}')
    print(f'\n~{tb/4.2/1e6:.0f}M tokens at 4.2 bytes/token')
    print(f"stats -> {args.stats}")


if __name__ == "__main__":
    main()
