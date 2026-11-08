#!/usr/bin/env python3
"""Validate a submission file against the competition test set.

Usage: python work/verify_sub.py path/to/submission.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = open(os.path.join(HERE, "test_names.txt")).read().split()


def main(path: str) -> int:
    df = pd.read_csv(path)
    problems = []
    if list(df.columns) != ["ID", "Emotion"]:
        problems.append(f"columns are {list(df.columns)}, expected ['ID','Emotion']")
    if len(df) != len(NAMES):
        problems.append(f"{len(df)} rows, expected {len(NAMES)}")
    ids = set(df["ID"].astype(str))
    missing = set(NAMES) - ids
    extra = ids - set(NAMES)
    if missing:
        problems.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    emo = df["Emotion"]
    if not np.issubdtype(emo.dtype, np.integer):
        problems.append(f"Emotion dtype is {emo.dtype}, expected integer")
    elif emo.min() < 0 or emo.max() > 5:
        problems.append(f"Emotion range [{emo.min()}, {emo.max()}], expected 0..5")
    dist = np.bincount(np.clip(emo.astype(int), 0, 5), minlength=6) / max(1, len(df))
    print(f"{path}: {len(df)} rows, label distribution " + ", ".join(f"{i}:{d:.3f}" for i, d in enumerate(dist)))
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: format matches the competition test set")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
