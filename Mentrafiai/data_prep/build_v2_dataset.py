"""
Build sft_train_v2.jsonl / sft_val_v2.jsonl from the cleaned originals plus the
augmentation preview. Inputs are read-only — originals and the preview stay on
disk untouched as backups.

Split policy: the preview is the only source of high-diversity examples, so a
held-out slice of it goes to val_v2; otherwise validation loss would still be
measured on the old 10-name distribution and would not reflect the new data.
De-duplicated on exact (user, assistant) text so the repeated communication /
finance-literacy examples cannot appear in both splits (that would leak).

Run:  python data_prep/build_v2_dataset.py
"""
import sys, os, json, random, hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

TRAIN_CLEAN = "data/processed/sft_train.clean.jsonl"
VAL_CLEAN = "data/processed/sft_val.clean.jsonl"
AUGMENT = "data/processed/sft_augment_preview.jsonl"
TRAIN_OUT = "data/processed/sft_train_v2.jsonl"
VAL_OUT = "data/processed/sft_val_v2.jsonl"
VAL_FRACTION = 0.10
SEED = 42


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def key(ex):
    return hashlib.sha1(
        (ex["user"] + "\x00" + ex["assistant"]).encode("utf-8")).hexdigest()


def main():
    for p in (TRAIN_CLEAN, VAL_CLEAN, AUGMENT):
        if not os.path.exists(p):
            print(f"MISSING input: {p}")
            return

    train_clean, val_clean, augment = load(TRAIN_CLEAN), load(VAL_CLEAN), load(AUGMENT)
    rng = random.Random(SEED)

    # Group augmentation examples by exact text so duplicates stay on one side.
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

    train = train_clean + aug_train
    val = val_clean + aug_val
    rng.shuffle(train); rng.shuffle(val)

    # Leakage guard: no exact example may appear in both splits.
    overlap = {key(e) for e in train} & {key(e) for e in val}
    assert not overlap, f"leak: {len(overlap)} examples in both splits"

    for path, rows in ((TRAIN_OUT, train), (VAL_OUT, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("inputs (left intact):")
    print(f"  {TRAIN_CLEAN:44} {len(train_clean):6}")
    print(f"  {VAL_CLEAN:44} {len(val_clean):6}")
    print(f"  {AUGMENT:44} {len(augment):6}  -> {len(aug_train)} train / {len(aug_val)} val")
    print("\noutputs:")
    print(f"  {TRAIN_OUT:44} {len(train):6}")
    print(f"  {VAL_OUT:44} {len(val):6}")
    print(f"\n  total {len(train) + len(val)} examples, "
          f"val share {100 * len(val) / (len(train) + len(val)):.1f}%, no split overlap")


if __name__ == "__main__":
    main()
