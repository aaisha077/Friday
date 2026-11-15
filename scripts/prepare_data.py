#!/usr/bin/env python3
"""Convert reviewed batches into MLX-LM training format and split train/valid/test.

Input : data/reviewed/*.jsonl  — Alpaca-style {"instruction", "input", "output"}
Output: data/mlx/{train,valid,test}.jsonl — chat format {"messages": [...]}

MLX-LM only accepts prompt/completion, messages (chat), or text formats — the raw
instruction/input/output shape is rejected. We use the chat format so training
matches how Llama-3.2-Instruct was originally tuned. The assistant turn is the
output object serialized to a JSON string, so the model learns to emit valid JSON.

The split is stratified by severity: with only ~18 critical examples, a plain
random split can leave the test set with zero criticals and make the eval useless.
"""

import glob
import json
import os
import random
from collections import defaultdict

SEED = 42
REVIEWED_GLOB = "data/reviewed/*.jsonl"
OUT_DIR = "data/mlx"
SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}  # test is the held-out set


def load_examples():
    examples = []
    for path in sorted(glob.glob(REVIEWED_GLOB)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    return examples


def to_chat(ex):
    """Alpaca {instruction,input,output} -> {"messages": [...]} chat record."""
    user = ex["instruction"].strip()
    if ex.get("input"):
        user += "\n\n" + ex["input"].strip()
    # Serialize the label object to a compact JSON string — this is the target
    # the model learns to generate.
    assistant = json.dumps(ex["output"], ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def stratified_split(examples):
    """Split proportionally within each severity so every split is representative."""
    by_sev = defaultdict(list)
    for ex in examples:
        by_sev[ex["output"]["severity"]].append(ex)

    buckets = {name: [] for name in SPLITS}
    rng = random.Random(SEED)
    for sev, items in sorted(by_sev.items()):
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * SPLITS["train"])
        n_valid = round(n * SPLITS["valid"])
        buckets["train"] += items[:n_train]
        buckets["valid"] += items[n_train:n_train + n_valid]
        buckets["test"] += items[n_train + n_valid:]

    for name in buckets:  # shuffle across severities so files aren't grouped
        rng.shuffle(buckets[name])
    return buckets


def main():
    examples = load_examples()
    buckets = stratified_split(examples)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loaded {len(examples)} reviewed examples\n")
    for name, items in buckets.items():
        out_path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for ex in items:
                f.write(json.dumps(to_chat(ex), ensure_ascii=False) + "\n")
        # per-split severity distribution, for a sanity check
        dist = defaultdict(int)
        for ex in items:
            dist[ex["output"]["severity"]] += 1
        dist_str = "  ".join(f"{k}:{dist[k]}" for k in ("low", "medium", "high", "critical"))
        print(f"{name:5s} {len(items):3d}  [{dist_str}]  -> {out_path}")


if __name__ == "__main__":
    main()
