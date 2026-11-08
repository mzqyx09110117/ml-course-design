import argparse
import os
import numpy as np
import torch
from common import load_split, predict_full, WORK
from eval_oof import load_model
from train import get_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(WORK, 'runs_v1'))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--out', default=os.path.join(WORK, 'pseudo_v1'))
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--hi', type=float, default=0.9)
    ap.add_argument('--lo', type=float, default=0.1)
    ap.add_argument('--tta', type=int, default=4)
    ap.add_argument('--cache', default='')
    ap.add_argument('--threads', type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    models = {}
    for f in range(5):
        p = os.path.join(args.run, f'f{f}_s{args.seed}', args.ckpt)
        m, ck = load_model(p)
        print('loaded fold', f, 'val', ck.get('val_dice'), flush=True)
        models[f] = m
    folds = get_folds()
    imgs, _, fovs = load_split(range(1, 21), 'test', cache=args.cache or None)
    for f in range(5):
        outdir = os.path.join(args.out, f'f{f}')
        os.makedirs(outdir, exist_ok=True)
        use = [g for g in range(5) if g != f]
        for i in range(1, 21):
            img = imgs[i].astype(np.float32)
            fov = fovs[i].astype(np.float32)
            acc = np.zeros(img.shape[1:], np.float32)
            for g in use:
                acc += predict_full(models[g], img, tta=args.tta)
            prob = acc / len(use) * fov
            label = (prob > args.thr).astype(np.uint8)
            valid = (((prob >= args.hi) | (prob <= args.lo)) & (fov > 0)).astype(np.uint8)
            np.savez_compressed(os.path.join(outdir, f'{i}.npz'), label=label, valid=valid,
                                prob=prob.astype(np.float16))
            print('fold %d img %d vessel %.4f valid %.4f' % (f, i, label.mean(), valid.mean()), flush=True)


if __name__ == '__main__':
    main()
