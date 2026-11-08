import argparse
import os
import numpy as np
from PIL import Image
from common import resize_to_512, WORK
from predict import rle_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ens-dir', default=os.path.join(WORK, 'submission_refined', 'ensprob'))
    ap.add_argument('--ref-dir', default=os.path.join(WORK, 'submission_refined', 'prob'))
    ap.add_argument('--w', type=float, default=0.5)
    ap.add_argument('--thr', type=float, default=0.45)
    ap.add_argument('--outdir', default=os.path.join(WORK, 'submission_final'))
    args = ap.parse_args()
    imgdir = os.path.join(args.outdir, 'image')
    os.makedirs(imgdir, exist_ok=True)
    rows = []
    for i in range(1, 21):
        ens = np.load(os.path.join(args.ens_dir, f'{i}.npy')).astype(np.float32)
        ref = np.load(os.path.join(args.ref_dir, f'{i}.npy')).astype(np.float32)
        p = args.w * ref + (1 - args.w) * ens
        mask = (p > args.thr).astype(np.uint8)
        Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(os.path.join(imgdir, f'{i}.png'))
        rows.append((str(i), rle_encode(resize_to_512(mask))))
        print('img %d frac %.4f' % (i, mask.mean()), flush=True)
    import pandas as pd
    pd.DataFrame(rows, columns=['Id', 'Predicted']).to_csv(os.path.join(args.outdir, 'submission.csv'), index=None)
    print('wrote', os.path.join(args.outdir, 'submission.csv'), flush=True)


if __name__ == '__main__':
    main()
