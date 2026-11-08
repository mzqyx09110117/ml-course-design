import numpy as np
import argparse
from common import load_split, resize_to_512, dice, WORK
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', nargs='+', default=[os.path.join(WORK, 'oof_best_03.npz'),
                                               os.path.join(WORK, 'oof_best_4.npz')])
    ap.add_argument('--b', default=os.path.join(WORK, 'oof_v1b.npz'))
    ap.add_argument('--out', default=os.path.join(WORK, 'oof_p2.npz'))
    args = ap.parse_args()
    A, cnt = {}, {}
    for p in args.a:
        d = np.load(p)
        for k in d.files:
            i = int(k)
            A[i] = A.get(i, 0) + d[k].astype(np.float32)
            cnt[i] = cnt.get(i, 0) + 1
    for i in A:
        A[i] /= cnt[i]
    B = {}
    d = np.load(args.b)
    for k in d.files:
        B[int(k)] = d[k].astype(np.float32)
    ids = sorted(set(A) & set(B))
    P = {i: (A[i] + B[i]) / 2 for i in ids}
    np.savez_compressed(args.out, **{str(k): v.astype(np.float16) for k, v in P.items()})
    _, labs, _ = load_split(range(1, 21), 'train')
    print('images', ids)
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55]:
        va = np.mean([dice(resize_to_512((A[i] > thr).astype(np.uint8)), resize_to_512(labs[i])) for i in ids])
        vb = np.mean([dice(resize_to_512((B[i] > thr).astype(np.uint8)), resize_to_512(labs[i])) for i in ids])
        vp = np.mean([dice(resize_to_512((P[i] > thr).astype(np.uint8)), resize_to_512(labs[i])) for i in ids])
        print('thr %.2f  A %.4f  B %.4f  p2 %.4f' % (thr, va, vb, vp))
    print('saved', args.out)


if __name__ == '__main__':
    main()
