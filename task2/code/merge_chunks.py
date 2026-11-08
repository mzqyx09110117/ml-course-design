#!/usr/bin/env python3
"""Merge fold-chunk outputs into one run directory.

Usage:
  python work/merge_chunks.py --out work/kaggle/V_merged/output/out_V \
      --srcdirs work/kaggle/V0/output2/out_V0 work/kaggle/V1/out2/out_V1 ...

Copies fold<k>_{oof,test}_{raw,ema}.npy + fold_ids.npy, merges the per-fold
summaries, and (if work/ytr.npy is present) prints the pooled OOF accuracy so
missing folds are obvious. Idempotent: re-running with more src dirs just adds
the new folds.
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--srcdirs", nargs="+", required=True)
    ap.add_argument("--y", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "work", "ytr.npy"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fold_ids_ref = None
    per_fold = {}
    for src in a.srcdirs:
        fi = np.load(os.path.join(src, "fold_ids.npy"))
        if fold_ids_ref is None:
            fold_ids_ref = fi
        else:
            assert np.array_equal(fold_ids_ref, fi), f"fold_ids mismatch: {src}"
        for f in glob.glob(os.path.join(src, "fold[0-9]_*.npy")):
            shutil.copy2(f, os.path.join(a.out, os.path.basename(f)))
        sp = os.path.join(src, "summary.json")
        if os.path.exists(sp):
            per_fold.update(json.load(open(sp)).get("per_fold", {}))
    np.save(os.path.join(a.out, "fold_ids.npy"), fold_ids_ref)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"merged_from": a.srcdirs, "per_fold": per_fold}, f, indent=2)

    for k, v in sorted(per_fold.items(), key=lambda kv: int(kv[0])):
        print(f"fold {k}: raw {v.get('oof_raw_acc')} ema {v.get('oof_ema_acc')}")

    if a.y and os.path.exists(a.y):
        y = np.load(a.y)
        for tag in ("raw", "ema"):
            pred = np.zeros((len(y), 6), np.float32)
            got = np.zeros(len(y), bool)
            for f in sorted(glob.glob(os.path.join(a.out, f"fold*_oof_{tag}.npy"))):
                k = int(re.search(r"fold(\d)", os.path.basename(f)).group(1))
                idx = np.where(fold_ids_ref == k)[0]
                arr = np.load(f)
                assert len(arr) == len(idx), (f, len(arr), len(idx))
                pred[idx] = arr
                got[idx] = True
            if got.all():
                print(f"OOF {tag}: {(pred.argmax(1) == y).mean():.4f} (all folds)")
            else:
                missing = sorted(set(fold_ids_ref[~got].tolist()))
                print(f"OOF {tag}: incomplete {got.sum()}/{len(y)}, missing folds {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
