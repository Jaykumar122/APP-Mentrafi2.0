"""
Fetch the v2 pretraining corpus: open-licensed sources only, one JSONL per source.

Why a v2 fetcher instead of extending fetch_pretrain_sources.py:
  * Every record carries {"text", "source", "domain"} so the achieved mix is
    auditable per-document. The v1 pipeline tracked nothing, which is how the old
    corpus ended up 0.0025% finance while data_mix.yaml claimed 25%.
  * Budgets are in BYTES and enforced while streaming, so a source cannot
    silently dominate (gutenberg was 41.6% of the old corpus) or silently vanish.
  * Quality-first substitutions: fineweb-edu-dedup replaces raw fineweb, and
    cosmopedia-v2 adds explanatory/textbook prose the old corpus had none of.

Licences (all explicitly open; verified against the HF dataset cards):
    HuggingFaceTB/smollm-corpus  fineweb-edu-dedup   ODC-By 1.0
    HuggingFaceTB/smollm-corpus  cosmopedia-v2       ODC-By 1.0
    wikimedia/wikipedia          20231101.en         CC-BY-SA 3.0 / GFDL
    Josephgflowers/Finance-Instruct-500k             Apache-2.0
    gbharti/finance-alpaca                           MIT
    sujet-ai/Sujet-Finance-Instruct-177k             Apache-2.0
    openai/gsm8k                                     MIT
    databricks/databricks-dolly-15k                  CC-BY-SA 3.0
    Project Gutenberg (sedthh/gutenberg_english)     public domain

Usage:
    python data_prep/fetch_corpus_v2.py                 # all sources, default budgets
    python data_prep/fetch_corpus_v2.py --only finance_instruct,gsm8k
    python data_prep/fetch_corpus_v2.py --list
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("data/raw/corpus_v2")
MIB = 1024 ** 2

# Domains exist so the mixer can reason about purpose, not just provenance.
#   general        broad English fluency
#   explanatory    textbook / step-by-step / "explain this" prose
#   encyclopedic   factual reference writing
#   finance        finance vocabulary, concepts, Q&A
#   mf_domain      mutual funds + personal finance specifically
#   numeracy       arithmetic / quantitative reasoning
#   instruction    following a request and answering it
#   literature     long-form narrative syntax

SOURCES = {}

# Sources that emit a second output file (see fetch_wikipedia) register its
# manifest record here so main() can fold it in.
_SIDECAR = {}


def source(name, domain, license_, budget_mib, note=""):
    def deco(fn):
        SOURCES[name] = {"name": name, "domain": domain, "license": license_,
                         "budget": budget_mib * MIB, "fn": fn, "note": note}
        return fn
    return deco


class Writer:
    """Append-only JSONL writer that stops itself at the byte budget.

    Tracks bytes of *text* (not JSON overhead) so budgets are comparable to the
    raw .txt sizes of the v1 corpus.
    """

    def __init__(self, path, source_name, domain, budget):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source, self.domain, self.budget = source_name, domain, budget
        self.text_bytes = 0
        self.n = 0
        self.t0 = time.time()
        self.fh = open(self.path, "w", encoding="utf-8")

    def add(self, text):
        text = (text or "").strip()
        if len(text) < 200:          # too short to teach anything at pretrain scale
            return True
        b = len(text.encode("utf-8"))
        self.fh.write(json.dumps(
            {"text": text, "source": self.source, "domain": self.domain},
            ensure_ascii=False) + "\n")
        self.text_bytes += b
        self.n += 1
        if self.n % 20000 == 0:
            self._progress()
        return self.text_bytes < self.budget

    def _progress(self):
        pct = 100 * self.text_bytes / self.budget
        rate = self.text_bytes / max(time.time() - self.t0, 1e-9) / MIB
        print(f"    {self.source:22} {self.n:>9,} docs  "
              f"{self.text_bytes/MIB:8.1f}/{self.budget/MIB:.0f} MiB  "
              f"({pct:5.1f}%)  {rate:5.2f} MiB/s", flush=True)

    def close(self):
        self.fh.close()
        self._progress()
        return {"source": self.source, "domain": self.domain,
                "docs": self.n, "text_bytes": self.text_bytes,
                "path": str(self.path), "seconds": round(time.time() - self.t0, 1)}


def _hf(repo, **kw):
    from datasets import load_dataset
    return load_dataset(repo, streaming=True, **kw)


def _shards(repo, prefix, n, revision=None):
    """Pick the first n parquet shards under prefix.

    Pointing at explicit files matters: letting `datasets` resolve a whole split
    makes it prefetch multiple multi-GB shards before .take() can stop it (the
    v1 script hit exactly this and pulled ~30GB for a 100k-row request).
    """
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    pq = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    return pq[:n]


# ---------------------------------------------------------------------------
# General English — educational web text
# ---------------------------------------------------------------------------
@source("fineweb_edu", "general", "ODC-By-1.0", 2048,
        "fineweb-edu-dedup: classifier-filtered educational web text")
def fetch_fineweb_edu(w):
    repo = "HuggingFaceTB/smollm-corpus"
    for f in _shards(repo, "fineweb-edu-dedup/", 24):
        print(f"      shard {f}", flush=True)
        for row in _hf(repo, data_files=f, split="train"):
            if not w.add(row.get("text")):
                return


# ---------------------------------------------------------------------------
# Explanatory / textbook prose — what the old corpus was missing entirely
# ---------------------------------------------------------------------------
@source("cosmopedia", "explanatory", "ODC-By-1.0", 1280,
        "cosmopedia-v2: synthetic textbook/explanatory/how-to prose")
def fetch_cosmopedia(w):
    repo = "HuggingFaceTB/smollm-corpus"
    for f in _shards(repo, "cosmopedia-v2/", 20):
        print(f"      shard {f}", flush=True)
        for row in _hf(repo, data_files=f, split="train"):
            if not w.add(row.get("text")):
                return


# ---------------------------------------------------------------------------
# Finance-specific Wikipedia — built by filtering the full dump rather than
# crawling, so it is reproducible and needs no scraper.
# ---------------------------------------------------------------------------
FIN_TERMS = re.compile(
    r"\b(mutual fund|expense ratio|net asset value|systematic investment plan|"
    r"asset allocation|portfolio|equity|debt security|bond yield|dividend|"
    r"capital gain|compound interest|inflation|interest rate|stock exchange|"
    r"index fund|exchange.traded fund|hedge fund|pension|annuity|risk tolerance|"
    r"diversification|securities|investor|valuation|balance sheet|cash flow|"
    r"amortization|derivative|volatility|Sharpe ratio|asset class|"
    r"personal finance|retirement planning|tax|credit|debt|savings|"
    r"stock market|bear market|bull market|liquidity|central bank|monetary policy)\b",
    re.I)


@source("wikipedia", "encyclopedic", "CC-BY-SA-3.0", 1280,
        "one pass over the dump, routed DISJOINTLY into finance vs general")
def fetch_wikipedia(w):
    """Route each article to exactly one of two output files in a single pass.

    Two things this fixes versus reusing v1's wikipedia.txt:
      * v1 chunked the dump at 8000 chars, shredding article boundaries; streaming
        fresh keeps one article per document.
      * A separate "finance subset" pull over the same dump would duplicate
        whatever the general pull already contained, and chunk-level hashing would
        not catch it. Disjoint routing makes overlap impossible by construction.

    The loop stops when the GENERAL budget fills, NOT when both fill. That is
    deliberate: probe_finance_density.py measured only 1.4% of Wikipedia bytes as
    finance-dense at this threshold, so waiting for a large finance sidecar to
    fill would mean streaming ~27 GB (more than the whole English dump) and
    throwing away the general text arriving alongside it. The sidecar budget is
    set above what a bounded pass can yield precisely so it never binds; whatever
    it collects is recorded in the manifest and the mixer works from that.
    """
    fin = Writer(OUT_DIR / "wikipedia_finance.jsonl", "wikipedia_finance",
                 "mf_domain", 512 * MIB)
    scanned = 0
    try:
        for row in _hf("wikimedia/wikipedia", name="20231101.en", split="train"):
            scanned += 1
            text = row.get("text") or ""
            if len(text) < 600:
                continue
            hits = {m.group(0).lower() for m in FIN_TERMS.finditer(text[:20000])}
            if len(hits) >= 4:
                fin.add(text)
            elif not w.add(text):
                break
        print(f"      scanned {scanned:,} articles; finance sidecar "
              f"{fin.text_bytes/MIB:.1f} MiB ({100*fin.text_bytes/max(w.text_bytes+fin.text_bytes,1):.1f}% "
              f"of streamed bytes)", flush=True)
    finally:
        rec = fin.close()
        rec["license"] = "CC-BY-SA-3.0"
        rec["note"] = "Wikipedia articles with >=4 distinct finance terms"
        _SIDECAR["wikipedia_finance"] = rec


# ---------------------------------------------------------------------------
# SEC 10-K filings — the primary source of real financial prose.
#
# Added after probe_finance_density.py established that general corpora are only
# ~1% finance by bytes, so the finance share cannot come from filtering general
# text. It has to come from genuinely financial documents, and US securities
# filings are public-domain government records with an Apache-2.0 dataset wrapper.
#
# Only the narrative sections are used:
#     1   Business            what the company does
#     1A  Risk Factors        risk articulation and hedged language
#     7   MD&A                management explaining results and outlook
#     7A  Market Risk         quantitative risk discussion
# The remaining items are stubs ("Not Applicable"), incorporation-by-reference,
# or raw statement tables, which the digit-heavy filter would drop anyway.
#
# The register is corporate and formal, which is a real risk to advisory tone if
# overweighted, so this is capped as a minority share of the final mix rather
# than allowed to dominate just because it is abundant.
# ---------------------------------------------------------------------------
EDGAR_SECTIONS = ("section_1A", "section_7", "section_7A", "section_1")
EDGAR_YEARS = ("2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013")
_EDGAR_STUB = re.compile(
    r"^\s*item\s+\d+[ab]?\.?[^\n]{0,120}\n\s*(not applicable|none|omitted|"
    r"reserved|not required)\b", re.I)

# Cap a single filing section at ~40k chars (~9.5k tokens, still ~9 full context
# windows). Measured on the first 1 GiB pull: mean 54,784 bytes/doc, p90 121,390,
# max 488,721 -- so 19,600 documents filled the entire budget and the top decile
# alone contributed 28k+ tokens each. Since the budget is in BYTES, a handful of
# enormous filings crowd out thousands of distinct companies, and repeated
# exposure to one company's boilerplate is worth far less than first exposure to
# another's. Truncating trades tail content for document diversity at identical
# byte cost: measured, it takes 19,600 docs/GiB to 36,671.
EDGAR_MAX_CHARS = 40000


def _trim(text, limit):
    """Truncate at the latest clean boundary before `limit`.

    The boundary search is a fallback chain because EDGAR has no paragraph
    structure: measured on the first pull, 0 of 10,289 over-length sections
    contained a blank line, so a paragraph-only search hard-cuts every one of
    them mid-sentence. Sentence-boundary search recovers them. The chain is kept
    rather than replaced because the other sources this may later be applied to
    do have paragraphs, and a paragraph break is the better cut when available.
    """
    if len(text) <= limit:
        return text
    for sep in ("\n\n", ". ", "\n", " "):
        cut = text.rfind(sep, limit // 2, limit)
        if cut > 0:
            return text[:cut + (len(sep) if sep == ". " else 0)].rstrip()
    return text[:limit]


@source("edgar", "finance", "Apache-2.0", 1024,
        "SEC 10-K narrative sections: Business, Risk Factors, MD&A, Market Risk")
def fetch_edgar(w):
    repo = "eloukas/edgar-corpus"
    rev = "refs/convert/parquet"          # the repo ships a loading script, which
                                          # `datasets` no longer executes; the Hub's
                                          # auto-converted parquet is the way in
    for year in EDGAR_YEARS:
        for f in _shards(repo, f"year_{year}/train/", 3, revision=rev):
            print(f"      shard {f}", flush=True)
            for row in _hf(repo, revision=rev, data_files=f, split="train"):
                for sec in EDGAR_SECTIONS:
                    text = row.get(sec) or ""
                    if len(text) < 1500 or _EDGAR_STUB.match(text):
                        continue
                    if not w.add(_trim(text, EDGAR_MAX_CHARS)):
                        return


# ---------------------------------------------------------------------------
# Finance Q&A / instruction — the knowledge the old pretrain never saw
# ---------------------------------------------------------------------------
def _joined(row, keys=("system", "instruction", "input", "question", "prompt",
                       "user", "context", "output", "response", "answer",
                       "assistant", "text")):
    """Flatten an instruction row into prose in a stable field order."""
    parts = []
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if not parts:
        parts = [str(v).strip() for v in row.values()
                 if isinstance(v, str) and v.strip()]
    return "\n\n".join(parts)


@source("finance_instruct", "finance", "Apache-2.0", 1536,
        "Finance-Instruct-500k: finance Q&A, reasoning and terminology")
def fetch_finance_instruct(w):
    for row in _hf("Josephgflowers/Finance-Instruct-500k", split="train"):
        if not w.add(_joined(row)):
            return


@source("finance_alpaca", "finance", "MIT", 128,
        "finance-alpaca: finance instruction/response pairs")
def fetch_finance_alpaca(w):
    for row in _hf("gbharti/finance-alpaca", split="train"):
        if not w.add(_joined(row)):
            return


@source("sujet_finance", "finance", "Apache-2.0", 384,
        "Sujet-Finance-Instruct-177k: finance tasks incl. QA and analysis")
def fetch_sujet_finance(w):
    for row in _hf("sujet-ai/Sujet-Finance-Instruct-177k", split="train"):
        if not w.add(_joined(row)):
            return


# ---------------------------------------------------------------------------
# Numeracy + instruction following
# ---------------------------------------------------------------------------
@source("gsm8k", "numeracy", "MIT", 32,
        "grade-school word problems with worked step-by-step solutions")
def fetch_gsm8k(w):
    for split in ("train", "test"):
        for row in _hf("openai/gsm8k", name="main", split=split):
            q, a = row.get("question", ""), row.get("answer", "")
            # Pad to clear the 200-char floor: short but genuinely useful items.
            if not w.add(f"Question: {q}\n\nStep-by-step solution: {a}"):
                return


@source("dolly", "instruction", "CC-BY-SA-3.0", 32,
        "human-written instruction/response pairs across categories")
def fetch_dolly(w):
    for row in _hf("databricks/databricks-dolly-15k", split="train"):
        if not w.add(_joined(row)):
            return


# ---------------------------------------------------------------------------
# Reused v1 sources — converted to the tracked JSONL format, and capped.
# Gutenberg is cut from 41.6% of the corpus to a few percent: 19th-century
# literary English is the style furthest from financial advisory prose, and it
# was consuming the largest single slice of the old compute budget.
# ---------------------------------------------------------------------------
def _reuse_txt(w, path, split_on="\n\n"):
    if not os.path.exists(path):
        print(f"      MISSING {path} — skipping", flush=True)
        return
    buf = []
    size = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            buf.append(line)
            size += len(line)
            if size > 8000:
                if not w.add("".join(buf)):
                    return
                buf, size = [], 0
    if buf:
        w.add("".join(buf))


# Project Gutenberg wraps every book in a header and a licence footer. The
# markers are stable across the collection, so the body is everything between
# them; without stripping, ~1-2 KiB of identical legal text per book becomes the
# most repeated passage in the whole source.
_PG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)

# A book is one row but must not be one document: the cleaner rejects anything
# over MAX_WORDS=100,000 as too_long, and a full novel exceeds that, so streaming
# whole books would silently discard most of this source. Chunks are accumulated
# on PARAGRAPH boundaries up to ~8,000 chars (~1,900 tokens, under the 1024-token
# context so no chunk is wasted) rather than cut at a fixed offset, because a
# mid-sentence cut is exactly the shredding this corpus criticises v1 for.
GUTENBERG_CHUNK_CHARS = 8000


@source("gutenberg", "literature", "public-domain", 256,
        "streamed from sedthh/gutenberg_english, capped hard "
        "(was 41.6% of the old corpus)")
def fetch_gutenberg(w):
    """Stream public-domain books, header-stripped and paragraph-chunked.

    Replaces the previous reuse of data/raw/pretrain/gutenberg/gutenberg.txt from
    v1. Streaming the dataset makes the source reproducible from the manifest
    alone instead of depending on a local artefact of an earlier pipeline -- which
    is not hypothetical: that local file was lost, and with it the only copy of
    this source's input.
    """
    repo = "sedthh/gutenberg_english"
    for f in _shards(repo, "data/", 2):
        print(f"      shard {f}", flush=True)
        for row in _hf(repo, data_files=f, split="train"):
            text = (row.get("TEXT") or "").replace("\r\n", "\n")
            m = _PG_START.search(text)
            if m:
                text = text[m.end():]
            m = _PG_END.search(text)
            if m:
                text = text[:m.start()]
            buf, size = [], 0
            for para in text.split("\n\n"):
                if not para.strip():
                    continue
                buf.append(para)
                size += len(para) + 2
                if size >= GUTENBERG_CHUNK_CHARS:
                    if not w.add("\n\n".join(buf)):
                        return
                    buf, size = [], 0
            if buf and not w.add("\n\n".join(buf)):
                return


@source("regulatory", "mf_domain", "CC-BY-SA-3.0", 8,
        "RBI/SEBI/AMFI explainer text carried over from v1 (tiny)")
def fetch_regulatory(w):
    for p in ("data/raw/pretrain/rbi/rbi.txt",
              "data/raw/pretrain/sebi/sebi.txt",
              "data/raw/pretrain/amfi/amfi.txt"):
        _reuse_txt(w, p)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of source names")
    ap.add_argument("--skip", default="", help="comma list of source names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="refetch even if output exists")
    ap.add_argument("--manifest", default="data/raw/corpus_v2/manifest.json")
    args = ap.parse_args()

    if args.list:
        print(f'{"source":22} {"domain":14} {"license":16} {"budget":>9}  note')
        print("-" * 108)
        tot = 0
        for s in SOURCES.values():
            tot += s["budget"]
            print(f'{s["name"]:22} {s["domain"]:14} {s["license"]:16} '
                  f'{s["budget"]/MIB:7.0f}M  {s["note"]}')
        print("-" * 108)
        print(f'{"TOTAL":22} {"":14} {"":16} {tot/MIB:7.0f}M'
              f'  (~{tot/4.2/1e6:.0f}M tokens at 4.2 bytes/token)')
        return

    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mpath = Path(args.manifest)
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    for name, s in SOURCES.items():
        if (only and name not in only) or name in skip:
            continue
        out = OUT_DIR / f"{name}.jsonl"
        if out.exists() and not args.force and name in manifest:
            print(f"[skip] {name}: already fetched "
                  f"({manifest[name]['text_bytes']/MIB:.1f} MiB)", flush=True)
            continue
        print(f"\n[fetch] {name}  domain={s['domain']}  license={s['license']}  "
              f"budget={s['budget']/MIB:.0f} MiB", flush=True)
        w = Writer(out, name, s["domain"], s["budget"])
        try:
            s["fn"](w)
        except KeyboardInterrupt:
            print("      interrupted — keeping what was written", flush=True)
        except Exception as e:
            print(f"      ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
        rec = w.close()
        rec["license"] = s["license"]
        rec["note"] = s["note"]
        manifest[name] = rec
        manifest.update(_SIDECAR)
        _SIDECAR.clear()
        mpath.write_text(json.dumps(manifest, indent=2))

    print("\n=== manifest ===")
    tot = sum(r["text_bytes"] for r in manifest.values())
    print(f'{"source":22} {"domain":14} {"docs":>10} {"MiB":>9} {"share":>7}')
    print("-" * 70)
    for r in sorted(manifest.values(), key=lambda r: -r["text_bytes"]):
        print(f'{r["source"]:22} {r["domain"]:14} {r["docs"]:10,} '
              f'{r["text_bytes"]/MIB:9.1f} {100*r["text_bytes"]/max(tot,1):6.1f}%')
    print("-" * 70)
    print(f'{"TOTAL":22} {"":14} {"":10} {tot/MIB:9.1f}  '
          f'(~{tot/4.2/1e6:.0f}M tokens at 4.2 bytes/token)')


if __name__ == "__main__":
    main()
    # HF streaming leaves prefetch threads that crash on interpreter teardown, so
    # exit hard — but os._exit skips buffer flushing, hence the explicit flush.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
