import argparse
import os
import numpy as np
import cv2
import torch
from PIL import Image
from common import load_split, predict_full, predict_full_ms, resize_to_512, UNet, WORK
from eval_oof import load_model, remove_small

cv2.setNumThreads(1)


def rle_encode(mask512):
    flat = mask512.reshape(-1)
    dots = np.where(flat == 1)[0]
    runs = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + 1, 0))
        runs[-1] += 1
        prev = b
    return ' '.join(str(r) for r in runs)


def resolve(p):
    if os.path.isdir(p):
        return os.path.join(p, 'best.pt')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--min-area', type=int, default=20)
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--scales', default='1.0')
    ap.add_argument('--cache', default='')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    outdir = args.outdir or os.path.join(WORK, 'submission_' + args.tag)
    imgdir = os.path.join(outdir, 'image')
    probdir = os.path.join(outdir, 'prob')
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(probdir, exist_ok=True)

    paths = [resolve(p) for p in args.ckpt]
    models = []
    for p in paths:
        m, ck = load_model(p)
        print('loaded', p, 'step', ck.get('step'), 'val', ck.get('val_dice'), flush=True)
        models.append(m)

    imgs, _, fovs = load_split(range(1, 21), 'test', cache=args.cache or None)
    scales = tuple(float(x) for x in args.scales.split(','))
    rows = []
    for i in range(1, 21):
        img = imgs[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        acc = np.zeros(img.shape[1:], np.float32)
        for m in models:
            acc += predict_full_ms(m, img, tta=bool(args.tta), scales=scales)
        prob = acc / len(models) * fov
        np.save(os.path.join(probdir, f'{i}.npy'), prob.astype(np.float16))
        mask = remove_small((prob > args.thr).astype(np.uint8), args.min_area)
        Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(os.path.join(imgdir, f'{i}.png'))
        m512 = resize_to_512(mask)
        rows.append((str(i), rle_encode(m512)))
        print('test %d vessel_frac %.4f' % (i, mask.mean()), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows, columns=['Id', 'Predicted'])
    csv = os.path.join(outdir, 'submission.csv')
    df.to_csv(csv, index=None)
    print('wrote', csv, flush=True)


if __name__ == '__main__':
    main()
