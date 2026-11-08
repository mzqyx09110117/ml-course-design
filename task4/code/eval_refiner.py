import argparse
import os
import numpy as np
import torch
from common import load_split, predict_full, resize_to_512, dice, adapt_bn, WORK
from eval_oof import load_model
from train import get_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(WORK, 'runs_refiner'))
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--oof', nargs='+', default=[os.path.join(WORK, 'oof_best_03.npz'),
                                                 os.path.join(WORK, 'oof_best_4.npz')])
    ap.add_argument('--cache', default='')
    ap.add_argument('--frangi', default='')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--adapt', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    imgs, labs, fovs = load_split(range(1, 21), 'train', cache=args.cache or None)
    oof = {}
    for p in args.oof:
        d = np.load(p)
        for k in d.files:
            oof[int(k)] = d[k].astype(np.float32)
    imgs6 = {}
    for i in range(1, 21):
        im = imgs[i].astype(np.float32)
        parts = [im[:4], oof[i][None]]
        if args.frangi:
            v = np.load(os.path.join(args.frangi, f'{i}.npy')).astype(np.float32)
            sel = fovs[i] > 0
            hi = np.percentile(v[sel], 99.5) + 1e-6
            parts.append(np.clip(v / hi, 0, 1)[None])
        parts.append(im[4:5])
        imgs6[i] = np.concatenate(parts, axis=0)
    folds = get_folds()
    all_probs = {}
    for f in [int(x) for x in args.folds.split(',') if x != '']:
        p = os.path.join(args.run, f'f{f}', args.ckpt)
        if not os.path.exists(p):
            print('missing', p, flush=True)
            continue
        model, ck = load_model(p)
        print('fold', f, 'step', ck.get('step'), flush=True)
        for i in folds[f]:
            if args.adapt:
                adapt_bn(model, [imgs6[i]])
            prob = predict_full(model, imgs6[i], tta=bool(args.tta)) * fovs[i].astype(np.float32)
            all_probs[i] = prob.astype(np.float16)
    if args.out:
        np.savez_compressed(args.out, **{str(k): v for k, v in all_probs.items()})
    ids = sorted(all_probs)
    print('ids', ids)
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        vals = [dice(resize_to_512((all_probs[i].astype(np.float32) > thr).astype(np.uint8)),
                     resize_to_512(labs[i])) for i in ids]
        print('thr %.2f  dice512 %.4f' % (thr, np.mean(vals)))
    vals = [dice(resize_to_512((all_probs[i].astype(np.float32) > 0.45).astype(np.uint8)),
                 resize_to_512(labs[i])) for i in ids]
    print('per-image @0.45:', [round(v, 3) for v in vals])


if __name__ == '__main__':
    main()
