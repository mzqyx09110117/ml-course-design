#!/usr/bin/env python3
"""Build a class-prior-balanced pseudo-label file for semi-supervised training.

Global confidence thresholds select mostly the majority/easy classes, which biases
the pseudo-labelled set.  Instead we take the most confident samples of every class
in proportion to the training prior, keeping a minimum confidence floor.

Usage: python work/pl_select.py --probs work/pl_probs.npy --budget 3500 --floor 0.6 \
                                --out work/pl_probs_topk.npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=os.path.join(HERE, "pl_probs.npy"))
    ap.add_argument("--y", default=os.path.join(HERE, "ytr.npy"))
    ap.add_argument("--budget", type=int, default=3500)
    ap.add_argument("--floor", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(HERE, "pl_probs_topk.npy"))
    a = ap.parse_args()

    q = np.load(a.probs).astype(np.float32)
    y = np.load(a.y)
    prior = np.bincount(y, minlength=q.shape[1]) / len(y)
    conf = q.max(1)
    lab = q.argmax(1)

    keep = []
    for c in range(q.shape[1]):
        n_c = int(round(a.budget * prior[c]))
        idx = np.where((lab == c) & (conf >= a.floor))[0]
        idx = idx[np.argsort(-conf[idx])][:n_c]
        keep.append(idx)
        if len(idx):
            print(f"class {c}: {len(idx):4d} pseudo-labels, conf range "
                  f"{conf[idx].min():.3f}..{conf[idx].max():.3f}")
        else:
            print(f"class {c}: none above the floor")
    keep = np.sort(np.concatenate(keep)) if keep else np.array([], int)

    out = np.zeros_like(q)
    out[keep] = q[keep]
    np.save(a.out, out)
    mix = np.bincount(q[keep].argmax(1), minlength=6) / max(1, len(keep))
    print(f"selected {len(keep)} samples ({len(keep)/len(q)*100:.1f}% of the test set)")
    print("pseudo-label class mix: " + ", ".join(f"{c}:{d:.3f}" for c, d in enumerate(mix)))
    print("train prior:            " + ", ".join(f"{c}:{d:.3f}" for c, d in enumerate(prior)))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
