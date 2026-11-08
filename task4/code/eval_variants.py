import argparse
import os
import numpy as np
import torch
from common import load_split, predict_full_ms, resize_to_512, dice, adapt_bn, WORK
from eval_oof import load_model
from train import get_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(WORK, 'runs_v1'))
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--scales', default='1.0')
    ap.add_argument('--adapt', default='none', choices=['none', 'set', 'per'])
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--cache', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    scales = tuple(float(x) for x in args.scales.split(','))
    imgs, labs, fovs = load_split(range(1, 21), 'train', cache=args.cache or None)
    folds = get_folds()
    use_folds = [int(x) for x in args.folds.split(',') if x != '']
    all_probs = {}
    for f in use_folds:
        p = os.path.join(args.run, f'f{f}_s1', args.ckpt)
        if not os.path.exists(p):
            print('missing', p)
            continue
        model, ck = load_model(p)
        vids = folds[f]
        if args.adapt == 'set':
            adapt_bn(model, [imgs[i].astype(np.float32) for i in vids])
        for i in vids:
            img = imgs[i].astype(np.float32)
            if args.adapt == 'per':
                adapt_bn(model, [img])
            prob = predict_full_ms(model, img, tta=bool(args.tta), scales=scales) * fovs[i].astype(np.float32)
            all_probs[i] = prob.astype(np.float16)
        print('fold', f, 'done', flush=True)
    if args.out:
        np.savez_compressed(args.out, **{str(k): v for k, v in all_probs.items()})
    ids = sorted(all_probs)
    print('thr   dice512')
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        vals = [dice(resize_to_512((all_probs[i].astype(np.float32) > thr).astype(np.uint8)),
                     resize_to_512(labs[i])) for i in ids]
        print('%.2f  %.4f' % (thr, np.mean(vals)))
    vals = [dice(resize_to_512((all_probs[i].astype(np.float32) > 0.5).astype(np.uint8)), resize_to_512(labs[i])) for i in ids]
    print('per-image @0.5:', [round(v, 3) for v in vals], 'mean %.4f' % np.mean(vals))


if __name__ == '__main__':
    main()
