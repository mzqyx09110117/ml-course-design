import argparse
import os
import numpy as np
import torch
from PIL import Image
from common import load_split, predict_full, resize_to_512, WORK
from eval_oof import load_model
from predict import rle_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refiner-run', nargs='+', default=[os.path.join(WORK, 'runs_refiner')])
    ap.add_argument('--ens-prob-dir', default=os.path.join(WORK, 'submission_ens_v1', 'prob'))
    ap.add_argument('--thr', type=float, default=0.45)
    ap.add_argument('--refiner-tta', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--outdir', default=os.path.join(WORK, 'submission_refined_ens'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    imgdir = os.path.join(args.outdir, 'image')
    probdir = os.path.join(args.outdir, 'prob')
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(probdir, exist_ok=True)

    refs = []
    for run in args.refiner_run:
        for f in range(5):
            p = os.path.join(run, f'f{f}', 'best.pt')
            if os.path.exists(p):
                m, _ = load_model(p)
                refs.append(m)
    print('refiners', len(refs), flush=True)
    imgs, _, fovs = load_split(range(1, 21), 'test')
    rows = []
    for i in range(1, 21):
        img = imgs[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        ens = np.load(os.path.join(args.ens_prob_dir, f'{i}.npy')).astype(np.float32) * fov
        six = np.concatenate([img[:4], ens[None], fov[None]], axis=0).astype(np.float32)
        acc = np.zeros(img.shape[1:], np.float32)
        for m in refs:
            acc += predict_full(m, six, tta=bool(args.refiner_tta))
        ref = acc / len(refs) * fov
        np.save(os.path.join(probdir, f'{i}.npy'), ref.astype(np.float16))
        mask = (ref > args.thr).astype(np.uint8)
        Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(os.path.join(imgdir, f'{i}.png'))
        rows.append((str(i), rle_encode(resize_to_512(mask))))
        print('test %d ens %.4f ref %.4f' % (i, (ens > args.thr).mean(), mask.mean()), flush=True)
    import pandas as pd
    pd.DataFrame(rows, columns=['Id', 'Predicted']).to_csv(os.path.join(args.outdir, 'submission.csv'), index=None)
    print('wrote', os.path.join(args.outdir, 'submission.csv'), flush=True)


if __name__ == '__main__':
    main()
