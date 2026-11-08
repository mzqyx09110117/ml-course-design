import argparse
import os
import numpy as np
import torch
from PIL import Image
from common import load_split, predict_full, resize_to_512, UNet, WORK
from eval_oof import load_model
from predict import rle_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-run', default=os.path.join(WORK, 'runs_v1'))
    ap.add_argument('--refiner-run', nargs='+', default=[os.path.join(WORK, 'runs_refiner')])
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--thr', type=float, default=0.45)
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--refiner-tta', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--outdir', default=os.path.join(WORK, 'submission_refined'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    outdir = args.outdir
    imgdir = os.path.join(outdir, 'image')
    probdir = os.path.join(outdir, 'prob')
    ensdir = os.path.join(outdir, 'ensprob')
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(probdir, exist_ok=True)
    os.makedirs(ensdir, exist_ok=True)

    folds = [int(x) for x in args.folds.split(',') if x != '']
    base_models, ref_models = {}, {}
    for run in args.refiner_run:
        for f in range(5):
            rp = os.path.join(run, f'f{f}', 'best.pt')
            if os.path.exists(rp):
                m, _ = load_model(rp)
                ref_models.setdefault(f, []).append(m)
    for f in range(5):
        bp = os.path.join(args.base_run, f'f{f}_s1', 'best.pt')
        if os.path.exists(bp) and f in ref_models:
            base_models[f], _ = load_model(bp)
    print('loaded', len(base_models), 'base and', sum(len(v) for v in ref_models.values()), 'refiners', flush=True)

    imgs, _, fovs = load_split(range(1, 21), 'test')
    rows = []
    for i in range(1, 21):
        img = imgs[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        base_acc = np.zeros(img.shape[1:], np.float32)
        ref_acc = np.zeros(img.shape[1:], np.float32)
        n = 0
        for f, bm in base_models.items():
            p = predict_full(bm, img, tta=bool(args.tta))
            base_acc += p
            six = np.concatenate([img[:4], p[None], fov[None]], axis=0).astype(np.float32)
            r = np.zeros_like(p)
            for rm in ref_models[f]:
                r += predict_full(rm, six, tta=bool(args.refiner_tta))
            ref_acc += r / len(ref_models[f])
            n += 1
        ens = base_acc / max(1, n) * fov
        ref = ref_acc / max(1, n) * fov
        np.save(os.path.join(ensdir, f'{i}.npy'), ens.astype(np.float16))
        np.save(os.path.join(probdir, f'{i}.npy'), ref.astype(np.float16))
        mask = (ref > args.thr).astype(np.uint8)
        Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(os.path.join(imgdir, f'{i}.png'))
        rows.append((str(i), rle_encode(resize_to_512(mask))))
        print('test %d base %.4f ref %.4f' % (i, (ens > args.thr).mean(), mask.mean()), flush=True)

    import pandas as pd
    pd.DataFrame(rows, columns=['Id', 'Predicted']).to_csv(os.path.join(outdir, 'submission.csv'), index=None)
    print('wrote', os.path.join(outdir, 'submission.csv'), flush=True)


if __name__ == '__main__':
    main()
