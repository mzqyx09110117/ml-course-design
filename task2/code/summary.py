#!/usr/bin/env python3
"""Collect per-run summaries (config + OOF accuracy) into one table."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Y = np.load(os.path.join(HERE, "ytr.npy"))


def main():
    rows = []
    for sj in sorted(glob.glob(os.path.join(HERE, "kaggle", "*", "output", "out_*", "summary.json"))):
        d = os.path.join(os.path.dirname(sj))
        try:
            s = json.load(open(sj))
        except Exception:
            continue
        args = s.get("args", {})
        folds = s.get("folds_done", [])
        oof_raw, oof_ema = [], []
        fold_ids = None
        p = os.path.join(d, "fold_ids.npy")
        if os.path.exists(p):
            fold_ids = np.load(p)
        for k in folds:
            for tag, store in (("raw", oof_raw), ("ema", oof_ema)):
                f = os.path.join(d, f"fold{k}_oof_{tag}.npy")
                if os.path.exists(f):
                    arr = np.load(f)
                    store.append((np.asarray(arr).argmax(1) == Y[fold_ids == k]).mean())
        rows.append({
            "run": os.path.basename(d).replace("out_", ""),
            "arch": args.get("arch"),
            "size": args.get("size"),
            "epochs": args.get("epochs"),
            "lr": args.get("lr"),
            "seed": args.get("seed"),
            "folds": len(folds),
            "raw": float(np.mean(oof_raw)) if oof_raw else None,
            "ema": float(np.mean(oof_ema)) if oof_ema else None,
            "min": round(s.get("elapsed_min", 0), 1),
        })
    hdr = f"{'run':<4}{'arch':<24}{'size':>5}{'ep':>5}{'lr':>9}{'seed':>5}{'folds':>6}{'oof_raw':>9}{'oof_ema':>9}{'min':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<4}{str(r['arch']):<24}{str(r['size']):>5}{str(r['epochs']):>5}"
              f"{('%.1e' % r['lr']) if r['lr'] else '-':>9}{str(r['seed']):>5}{r['folds']:>6}"
              f"{('%.4f' % r['raw']) if r['raw'] else '-':>9}{('%.4f' % r['ema']) if r['ema'] else '-':>9}{r['min']:>7}")
    return rows


if __name__ == "__main__":
    main()
