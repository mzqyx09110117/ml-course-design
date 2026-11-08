import argparse
import os
import numpy as np
import cv2
import torch
from common import load_split, predict_full, predict_full_ms, resize_to_512, dice, UNet, UNetV2, MTLUNet, WORK
from train import get_folds

cv2.setNumThreads(1)


def _conv_key(sd):
    return 'enc.0.block.0.weight' if 'enc.0.block.0.weight' in sd else 'enc.0.c1.weight'


def model_base(sd):
    return sd[_conv_key(sd)].shape[0]


def model_inch(sd):
    return sd[_conv_key(sd)].shape[1]


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    arch = ck.get('arch', 'v1')
    cls = {'v1': UNet, 'v2': UNetV2, 'mtl': MTLUNet}.get(arch, UNet)
    m = cls(in_ch=model_inch(ck['model']), base=ck.get('base') or model_base(ck['model']))
    m.load_state_dict(ck['model'])
    m.eval()
    return m, ck


def remove_small(mask, min_area):
    if min_area <= 0:
        return mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_area:
            keep[k] = 1
    return keep[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(WORK, 'runs_v1'))
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--scales', default='1.0')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--split-seed', type=int, default=123)
    ap.add_argument('--cache', default='')
    ap.add_argument('--out', default=os.path.join(WORK, 'oof_v1.npz'))
    ap.add_argument('--threads', type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    imgs, labs, fovs = load_split(range(1, 21), 'train', cache=args.cache or None)
    folds = get_folds(args.split_seed)
    scales = tuple(float(x) for x in args.scales.split(','))
    probs = {}
    fold_list = [int(x) for x in args.folds.split(',') if x != '']
    for f in fold_list:
        p = os.path.join(args.run, f'f{f}_s{args.seed}', args.ckpt)
        if not os.path.exists(p):
            print('missing', p, flush=True)
            continue
        model, ck = load_model(p)
        print('fold', f, 'step', ck.get('step'), 'val', ck.get('val_dice'), flush=True)
        for i in folds[f]:
            prob = predict_full_ms(model, imgs[i].astype(np.float32), tta=bool(args.tta),
                                   scales=scales) * fovs[i].astype(np.float32)
            probs[i] = prob.astype(np.float16)
    np.savez_compressed(args.out, **{str(k): v for k, v in probs.items()})
    print('saved', args.out, 'images', len(probs), flush=True)

    ids = sorted(probs)
    if not ids:
        return
    print('thr  minarea  dice_raw  dice_512')
    for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        for ma in [0, 10, 20, 40]:
            raw, d512 = [], []
            for i in ids:
                prob = probs[i].astype(np.float32)
                m = (prob > thr).astype(np.uint8)
                m = remove_small(m, ma)
                raw.append(dice(m, labs[i]))
                d512.append(dice(resize_to_512(m), resize_to_512(labs[i])))
            print('%.2f %6d   %.4f    %.4f' % (thr, ma, np.mean(raw), np.mean(d512)), flush=True)


if __name__ == '__main__':
    main()
