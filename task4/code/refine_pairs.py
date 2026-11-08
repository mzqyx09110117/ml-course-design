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
    ap.add_argument('--run-a', default=os.path.join(WORK, 'runs_v1'))
    ap.add_argument('--run-b', default=os.path.join(WORK, 'runs_v1b'))
    ap.add_argument('--refiner-run', default=os.path.join(WORK, 'runs_refiner_pp'))
    ap.add_argument('--thr', type=float, default=0.45)
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--refiner-tta', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--outdir', default=os.path.join(WORK, 'submission_pairs'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    outdir = args.outdir
    imgdir = os.path.join(outdir, 'image')
    probdir = os.path.join(outdir, 'prob')
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(probdir, exist_ok=True)

    A, B, R = {}, {}, {}
    for f in range(5):
        pa = os.path.join(args.run_a, f'f{f}_s1', 'best.pt')
        pb = os.path.join(args.run_b, f'f{f}_s1', 'best.pt')
        pr = os.path.join(args.refiner_run, f'f{f}', 'best.pt')
        if os.path.exists(pa) and os.path.exists(pb) and os.path.exists(pr):
            A[f], _ = load_model(pa)
            B[f], _ = load_model(pb)
            R[f], _ = load_model(pr)
    print('loaded folds', sorted(A), flush=True)

    imgs, _, fovs = load_split(range(1, 21), 'test')
    rows = []
    for i in range(1, 21):
        img = imgs[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        acc = np.zeros(img.shape[1:], np.float32)
        n = 0
        for f in sorted(R):
            pA = predict_full(A[f], img, tta=bool(args.tta))
            pB = predict_full(B[f], img, tta=bool(args.tta))
            p2 = (pA + pB) / 2
            six = np.concatenate([img[:4], p2[None], fov[None]], axis=0).astype(np.float32)
            acc += predict_full(R[f], six, tta=bool(args.refiner_tta))
            n += 1
        ref = acc / max(1, n) * fov
        np.save(os.path.join(probdir, f'{i}.npy'), ref.astype(np.float16))
        mask = (ref > args.thr).astype(np.uint8)
        Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(os.path.join(imgdir, f'{i}.png'))
        rows.append((str(i), rle_encode(resize_to_512(mask))))
        print('test %d frac %.4f' % (i, mask.mean()), flush=True)
    import pandas as pd
    pd.DataFrame(rows, columns=['Id', 'Predicted']).to_csv(os.path.join(outdir, 'submission.csv'), index=None)
    print('wrote', os.path.join(outdir, 'submission.csv'), flush=True)


if __name__ == '__main__':
    main()
