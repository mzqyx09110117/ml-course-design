import argparse
import os
import numpy as np
from common import load_split, resize_to_512, dice, WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', nargs='+', default=[os.path.join(WORK, 'oof_best_03.npz'),
                                                  os.path.join(WORK, 'oof_best_4.npz')])
    ap.add_argument('--ref', default=os.path.join(WORK, 'oof_refiner_all.npz'))
    args = ap.parse_args()
    _, labs, _ = load_split(range(1, 21), 'train')
    base, ref, cnt = {}, {}, {}
    for p in args.base:
        d = np.load(p)
        for k in d.files:
            i = int(k)
            base[i] = base.get(i, 0) + d[k].astype(np.float32)
            cnt[i] = cnt.get(i, 0) + 1
    for i in base:
        base[i] /= cnt[i]
    d = np.load(args.ref)
    for k in d.files:
        ref[int(k)] = d[k].astype(np.float32)
    ids = sorted(set(base) & set(ref))
    gt = {i: resize_to_512(labs[i]) for i in ids}
    print('images', len(ids))
    print('w      thr0.35 thr0.40 thr0.45 thr0.50 thr0.55')
    best = (-1, None, None)
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        line = []
        for thr in [0.35, 0.40, 0.45, 0.50, 0.55]:
            vals = []
            for i in ids:
                p = w * ref[i] + (1 - w) * base[i]
                vals.append(dice(resize_to_512((p > thr).astype(np.uint8)), gt[i]))
            m = float(np.mean(vals))
            line.append('%.4f' % m)
            if m > best[0]:
                best = (m, w, thr)
        print('%.2f   %s' % (w, '  '.join(line)))
    print('BEST mean %.4f w=%.2f thr=%.2f' % best)
    w, thr = best[1], best[2]
    per = []
    for i in ids:
        p = w * ref[i] + (1 - w) * base[i]
        per.append(dice(resize_to_512((p > thr).astype(np.uint8)), gt[i]))
    print('per-image:', [round(v, 3) for v in per])


if __name__ == '__main__':
    main()
