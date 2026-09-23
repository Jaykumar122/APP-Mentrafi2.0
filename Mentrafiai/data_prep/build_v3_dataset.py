"""
Build sft_train_v3.jsonl / sft_val_v3.jsonl from the v2 dataset plus the v3
augmentation preview. Inputs are read-only — v2 files and the preview stay on
disk untouched as backups.

Split policy (mirrors build_v2_dataset.py): the existing v2 train/val splits are
carried over as-is; the NEW v3 preview is the only fresh source, so a held-out
~10% slice of it goes to val_v3 (otherwise val would not reflect the new short/
name-fidelity data). Grouped by exact (user, assistant) text so identical
examples never straddle the split, then a hard leakage guard asserts no exact
example appears in both final splits. Splits are also de-duplicated on exact
text so a v3 example that happens to equal a v2 one isn't double-counted.

Run:  python data_prep/build_v3_dataset.py
"""
import sys, os, json, random, hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

TRAIN_V2 = "data/processed/sft_train_v2.jsonl"
VAL_V2 = "data/processed/sft_val_v2.jsonl"
AUGMENT = "data/processed/sft_augment_v3_preview.jsonl"
TRAIN_OUT = "data/processed/sft_train_v3.jsonl"
VAL_OUT = "data/processed/sft_val_v3.jsonl"
VAL_FRACTION = 0.10
SEED = 43


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def key(ex):
    return hashlib.sha1((ex["user"] + "\x00" + ex["assistant"]).encode("utf-8")).hexdigest()


def dedupe(rows):
    seen, out = set(), []
    for ex in rows:
        k = key(ex)
        if k in seen:
            continue
        seen.add(k); out.append(ex)
    return out


def main():
    for p in (TRAIN_V2, VAL_V2, AUGMENT):
        if not os.path.exists(p):
            print(f"MISSING input: {p}")
            return

    train_v2, val_v2, augment = load(TRAIN_V2), load(VAL_V2), load(AUGMENT)
    rng = random.Random(SEED)

    # Group the new v3 examples by exact text so duplicates stay on one side.
    groups = {}
    for ex in augment:
        groups.setdefault(key(ex), []).append(ex)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)

    target_val = int(round(len(augment) * VAL_FRACTION))
    aug_val, aug_train, n_val = [], [], 0
    for k in group_keys:
        if n_val < target_val:
            aug_val.extend(groups[k]); n_val += len(groups[k])
        else:
            aug_train.extend(groups[k])

    train = dedupe(train_v2 + aug_train)
    val = dedupe(val_v2 + aug_val)
    rng.shuffle(train); rng.shuffle(val)

    # Leakage guard: no exact example may appear in both splits. If a v3 val
    # example collides with a v2 train example, drop it from val (train wins).
    train_keys = {key(e) for e in train}
    val = [e for e in val if key(e) not in train_keys]
    overlap = train_keys & {key(e) for e in val}
    assert not overlap, f"leak: {len(overlap)} examples in both splits"

    for path, rows in ((TRAIN_OUT, train), (VAL_OUT, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("inputs (left intact):")
    print(f"  {TRAIN_V2:44} {len(train_v2):6}")
    print(f"  {VAL_V2:44} {len(val_v2):6}")
    print(f"  {AUGMENT:44} {len(augment):6}  -> {len(aug_train)} train / {len(aug_val)} val")
    print("\noutputs:")
    print(f"  {TRAIN_OUT:44} {len(train):6}")
    print(f"  {VAL_OUT:44} {len(val):6}")
    print(f"\n  total {len(train) + len(val)} examples, "
          f"val share {100 * len(val) / (len(train) + len(val)):.1f}%, no split overlap")


if __name__ == "__main__":
    main()
