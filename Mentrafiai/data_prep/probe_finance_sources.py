"""
Probe candidate finance datasets for availability, licence and usable text size.

Written because the batch-1 fetch established that the finance sources already in
fetch_corpus_v2.py yield only ~586 MiB of usable text. Reaching a meaningful
finance share against a multi-GB general corpus therefore needs either more
sources or aggressive upsampling, and more sources is strictly better.

This probe does NOT download bulk data. It streams a handful of rows per
candidate, so a repo that is gated, renamed, restructured or wrong-licensed is
discovered in seconds instead of thirty minutes into a fetch.

Usage:
    python data_prep/probe_finance_sources.py
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (repo, config_name_or_None, split, text_fields)
CANDIDATES = [
    # US SEC filings: public-domain government documents, real disclosure prose.
    ("eloukas/edgar-corpus", "year_2020", "train", ["section_1", "section_7", "full_text"]),
    ("eloukas/edgar-corpus", "full", "train", ["section_1", "section_7"]),
    # Financial numerical reasoning over tables+text.
    ("dreamproit/finqa", None, "train", ["question", "answer", "pre_text"]),
    ("ibm-research/finqa", None, "train", ["question", "answer", "pre_text"]),
    # Financial news / earnings-call style prose.
    ("ashraq/financial-news-articles", None, "train", ["title", "text"]),
    ("BEE-spoke-data/fineweb-finance", None, "train", ["text"]),
    # Domain-adaptive pretraining corpora assembled by others.
    ("AdaptLLM/finance-tasks", "ConvFinQA", "test", ["input", "options"]),
    ("FinLang/investopedia-embedding-dataset", None, "train", ["text"]),
    ("gbharti/wealth-alpaca_lora", None, "train", ["instruction", "input", "output"]),
    # Personal-finance discussion (register closest to an advisory conversation).
    ("nihiluis/financial-advisor-100k", None, "train", ["question", "answer"]),
    ("Malikeh1375/medical-question-answering-datasets", None, "train", ["input"]),
]


def card_license(repo):
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo)
        tags = [t for t in (info.tags or []) if t.startswith("license:")]
        if tags:
            return ",".join(t.split(":", 1)[1] for t in tags)
        cd = getattr(info, "cardData", None) or {}
        lic = cd.get("license")
        if isinstance(lic, list):
            return ",".join(lic)
        return lic or "?"
    except Exception as e:
        return f"<{type(e).__name__}>"


def probe(repo, name, split, fields, n=3):
    from datasets import load_dataset
    kw = {"streaming": True, "split": split}
    if name:
        kw["name"] = name
    ds = load_dataset(repo, **kw)
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i + 1 >= n:
            break
    if not rows:
        return {"ok": False, "why": "no rows"}
    keys = list(rows[0].keys())
    present = [f for f in fields if f in keys]
    use = present or [k for k, v in rows[0].items() if isinstance(v, str)]
    sizes = []
    for r in rows:
        text = "\n\n".join(str(r.get(k, "")) for k in use if r.get(k))
        sizes.append(len(text))
    sample = "\n\n".join(str(rows[0].get(k, "")) for k in use if rows[0].get(k))
    return {"ok": True, "keys": keys, "text_fields": use,
            "mean_chars": sum(sizes) // len(sizes),
            "sample": sample[:300].replace("\n", " ")}


def main():
    results = []
    for repo, name, split, fields in CANDIDATES:
        tag = f"{repo}" + (f":{name}" if name else "")
        print(f"\n--- {tag}  split={split}", flush=True)
        lic = card_license(repo)
        print(f"    license: {lic}", flush=True)
        rec = {"repo": repo, "name": name, "split": split, "license": lic}
        try:
            r = probe(repo, name, split, fields)
            rec.update(r)
            if r["ok"]:
                print(f"    OK  fields={r['text_fields']}  "
                      f"mean {r['mean_chars']:,} chars/row", flush=True)
                print(f"    sample: {r['sample'][:200]}", flush=True)
            else:
                print(f"    EMPTY: {r['why']}", flush=True)
        except Exception as e:
            rec.update({"ok": False, "why": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"    FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
        results.append(rec)

    out = Path("data/raw/corpus_v2/probe_finance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 92)
    print(f'{"dataset":48} {"license":16} {"chars/row":>10}  status')
    print("-" * 92)
    for r in results:
        tag = (r["repo"] + (f":{r['name']}" if r["name"] else ""))[:47]
        if r.get("ok"):
            print(f'{tag:48} {str(r["license"])[:15]:16} {r["mean_chars"]:10,}  usable')
        else:
            print(f'{tag:48} {str(r["license"])[:15]:16} {"":>10}  {r.get("why","")[:28]}')
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import os
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
