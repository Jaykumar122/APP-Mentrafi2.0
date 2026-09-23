"""
Measure how much of each general corpus is finance-dense enough to route.

The plan is to extend fetch_wikipedia's disjoint-routing trick to fineweb-edu and
cosmopedia: one pass over the stream, finance-dense documents into a finance
sidecar, everything else into the general file. That costs no extra download and
no extra licence risk, which makes it strictly better than adding a
questionably-licensed finance dataset.

But the routing yield decides whether it is worth anything. If 4% of fineweb-edu
is finance-dense, filling a 512 MiB sidecar needs ~12 GB of streaming; if it is
15%, it needs 3.4 GB. Budgets set by guesswork here reproduce exactly the failure
this rebuild exists to fix (data_mix.yaml claimed 25% finance, the corpus
delivered 0.0025%), so the thresholds and budgets come from this measurement.

Reports, per source and per threshold, the share of DOCUMENTS and the share of
BYTES that would route to finance. Bytes is the number that matters for budgets.

Usage:
    python data_prep/probe_finance_density.py --docs 4000
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_prep.fetch_corpus_v2 import FIN_TERMS, _hf, _shards

THRESHOLDS = (3, 4, 5, 6, 8, 10)


def distinct_terms(text, window=20000):
    return len({m.group(0).lower() for m in FIN_TERMS.finditer(text[:window])})


def measure(name, rows, min_chars, docs_target):
    """Return {threshold: (doc_share, byte_share)} plus totals."""
    hits = {t: [0, 0] for t in THRESHOLDS}       # [docs, bytes]
    n = tot_bytes = 0
    examples = {t: None for t in THRESHOLDS}
    for row in rows:
        text = row.get("text") or ""
        if len(text) < min_chars:
            continue
        b = len(text.encode("utf-8"))
        n += 1
        tot_bytes += b
        d = distinct_terms(text)
        for t in THRESHOLDS:
            if d >= t:
                hits[t][0] += 1
                hits[t][1] += b
                if examples[t] is None and t >= 6:
                    examples[t] = text[:200].replace("\n", " ")
        if n >= docs_target:
            break
    return {"source": name, "docs": n, "bytes": tot_bytes,
            "mean_doc_bytes": tot_bytes // max(n, 1),
            "shares": {t: {"doc_share": hits[t][0] / max(n, 1),
                           "byte_share": hits[t][1] / max(tot_bytes, 1)}
                       for t in THRESHOLDS},
            "example_at_6": examples.get(6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=4000)
    ap.add_argument("--out", default="data/raw/corpus_v2/probe_density.json")
    args = ap.parse_args()

    smol = "HuggingFaceTB/smollm-corpus"
    plans = []
    try:
        fw = _shards(smol, "fineweb-edu-dedup/", 1)[0]
        plans.append(("fineweb_edu", lambda f=fw: _hf(smol, data_files=f, split="train"), 600))
    except Exception as e:
        print(f"fineweb shard lookup failed: {e}")
    try:
        cp = _shards(smol, "cosmopedia-v2/", 1)[0]
        plans.append(("cosmopedia", lambda f=cp: _hf(smol, data_files=f, split="train"), 600))
    except Exception as e:
        print(f"cosmopedia shard lookup failed: {e}")
    plans.append(("wikipedia",
                  lambda: _hf("wikimedia/wikipedia", name="20231101.en", split="train"), 600))

    results = []
    for name, loader, min_chars in plans:
        print(f"\n--- {name}: sampling {args.docs:,} docs (min {min_chars} chars)",
              flush=True)
        try:
            r = measure(name, loader(), min_chars, args.docs)
        except Exception as e:
            print(f"    FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        results.append(r)
        print(f"    {r['docs']:,} docs, {r['bytes']/1024**2:.1f} MiB, "
              f"mean {r['mean_doc_bytes']:,} B/doc", flush=True)
        for t in THRESHOLDS:
            s = r["shares"][t]
            print(f"      >={t:2} distinct terms: {s['doc_share']*100:5.1f}% of docs, "
                  f"{s['byte_share']*100:5.1f}% of bytes", flush=True)

    if not results:
        print("nothing measured"); return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 84)
    print("BYTE share routing to finance, by threshold")
    print(f'{"source":14}' + "".join(f'{">="+str(t):>10}' for t in THRESHOLDS))
    print("-" * 84)
    for r in results:
        print(f'{r["source"]:14}' + "".join(
            f'{r["shares"][t]["byte_share"]*100:9.1f}%' for t in THRESHOLDS))
    print("-" * 84)
    print("\nTo fill a finance sidecar of S MiB from a source with byte-share p,")
    print("the stream has to deliver S/p MiB of text. Read the general budget off")
    print("the same row: general gets (1-p) of whatever is streamed.")
    for r in results:
        for t in (5, 6):
            p = r["shares"][t]["byte_share"]
            if p > 0:
                print(f'  {r["source"]:12} >={t}: 256 MiB finance needs '
                      f'{256/p:7.0f} MiB streamed ({256/p*(1-p):7.0f} MiB to general)')
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
