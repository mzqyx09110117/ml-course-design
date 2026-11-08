import argparse
import os
import numpy as np
import cv2
from PIL import Image
from common import load_split, resize_to_512, dice, WORK


def prob_to_512_cv(prob, size=512):
    return cv2.resize(prob.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def mask_to_512_nearest(mask, size=512):
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', nargs='+', default=[os.path.join(WORK, 'oof_f0.npz')])
    ap.add_argument('--min-area', type=int, default=0)
    args = ap.parse_args()
    _, labs, _ = load_split(range(1, 21), 'train')
    probs, counts = {}, {}
    for p in args.oof:
        d = np.load(p)
        for k in d.files:
            i = int(k)
            probs[i] = probs.get(i, 0) + d[k].astype(np.float32)
            counts[i] = counts.get(i, 0) + 1
    for i in probs:
        probs[i] = probs[i] / counts[i]
    ids = sorted(probs)
    gt512 = {i: resize_to_512(labs[i]) for i in ids}
    print('images', ids)
    print('thr   A_fullthr_resize   B_prob512_thr   C_fullthr_nearest')
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        a, b, c = [], [], []
        for i in ids:
            p = probs[i]
            ma = resize_to_512((p > thr).astype(np.uint8))
            mb = (prob_to_512_cv(p) > thr).astype(np.uint8)
            mc = mask_to_512_nearest((p > thr).astype(np.uint8))
            a.append(dice(ma, gt512[i]))
            b.append(dice(mb, gt512[i]))
            c.append(dice(mc, gt512[i]))
        print('%.2f      %.4f            %.4f          %.4f' % (thr, np.mean(a), np.mean(b), np.mean(c)))
    if args.min_area > 0:
        for thr in [0.4, 0.45, 0.5]:
            vals = []
            for i in ids:
                m = ((probs[i] > thr).astype(np.uint8))
                n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
                keep = np.zeros(n, np.uint8)
                for k in range(1, n):
                    if st[k, cv2.CC_STAT_AREA] >= args.min_area:
                        keep[k] = 1
                vals.append(dice(resize_to_512(keep[lab]), gt512[i]))
            print('minarea', args.min_area, 'thr', thr, 'A dice %.4f' % np.mean(vals))
    per = []
    for i in ids:
        per.append(dice(resize_to_512((probs[i] > 0.5).astype(np.uint8)), gt512[i]))
    print('per-image @0.5 A:', [round(v, 3) for v in per])


if __name__ == '__main__':
    main()
