import argparse, os, glob
import numpy as np
import cv2
import pandas as pd
import torch

from fovea_lib import DATA, FoveaNet, heatmap_peak, normalize, IMG_MEAN, IMG_STD


def load_model(ckpt_path, device='cpu'):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck['args']
    in_ch = 4 if a.get('disc_ch', 0) else 3
    model = FoveaNet(a['backbone'], pretrained=False, freeze_stem=bool(a.get('freeze_stem', 1)),
                     width=a['width'], in_ch=in_ch, stride2=bool(a.get('stride2', 0)))
    model.load_state_dict(ck['model'])
    model.eval()
    a['in_ch'] = in_ch
    return model, a



def predict_one(model, img1024, size=512, tta_flip=True, scales=(1.0,), disc=None, offset=False):
    """img1024 uint8 RGB square; returns normalised (ux, uy)."""
    W = img1024.shape[1]
    outs = []
    for sc in scales:
        s = max(64, int(round(size * sc / 32)) * 32)  # input must be divisible by 32
        im = cv2.resize(img1024, (s, s), interpolation=cv2.INTER_AREA)
        arr = normalize(im / 255.0)
        if getattr(model, 'stem', None) is not None and model.stem[0].in_channels == 4:
            from fovea_lib import disc_channel
            d = disc if disc is not None else (0.5, 0.5)
            arr = np.concatenate([arr, disc_channel(d[0] * s, d[1] * s, s,
                                                     sigma=10.0 * s / 512.0)[None]], 0)
        x = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            o = model(x)
            p = torch.softmax(o.flatten(1), dim=1).view_as(o)
            px, py = heatmap_peak(p)
            hm = o.shape[-1]
            ux, uy = (px + 0.5) / hm, (py + 0.5) / hm
            if offset and disc is not None:
                ux, uy = disc[0] + (ux - 0.5), disc[1] + (uy - 0.5)
            outs.append((ux, uy))
            if tta_flip:
                o2 = model(torch.flip(x, dims=[3]))
                p2 = torch.softmax(o2.flatten(1), dim=1).view_as(o2)
                px2, py2 = heatmap_peak(p2)
                fx2, fy2 = 1.0 - (px2 + 0.5) / hm, (py2 + 0.5) / hm
                if offset and disc is not None:
                    fx2, fy2 = disc[0] + (fx2 - 0.5), disc[1] + (fy2 - 0.5)
                outs.append((fx2, fy2))
    return float(np.mean([o[0] for o in outs])), float(np.mean([o[1] for o in outs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--no_flip', action='store_true')
    ap.add_argument('--scales', default='1.0')
    ap.add_argument('--agg', default='median', choices=['mean', 'median'])
    ap.add_argument('--save_raw', default='')
    args = ap.parse_args()

    torch.set_num_threads(12)
    scales = [float(s) for s in args.scales.split(',')]
    models = [load_model(p) for p in args.ckpts]  # (model, args) pairs
    discs = {}
    if any(ma.get('in_ch') == 4 for _, ma in models):
        _d = pd.read_csv('/home/mzqyx09110117/桌面/test3/fovea_work/disc_centers.csv', dtype={'id': str})
        discs = {r.id: (float(r.dx), float(r.dy)) for r in _d.itertuples()}
    print(f'{len(models)} models, scales={scales}, flip_tta={not args.no_flip}')

    if args.split == 'test':
        files = sorted(glob.glob(f'{DATA}/test/*.jpg'))
        ids = [os.path.basename(f).split('.')[0] for f in files]
    else:
        df = pd.read_csv(f'{DATA}/fovea_localization_train_GT.csv', dtype={'data': str})
        ids = [str(int(d)).zfill(4) for d in df.data]
        files = [f'{DATA}/train/{n}.jpg' for n in ids]

    rows = []
    for n, f in zip(ids, files):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        im1024 = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
        preds = []
        for m, ma in models:
            preds.append(predict_one(m, im1024, size=ma.get('size', 512),
                                     tta_flip=not args.no_flip, scales=scales,
                                     disc=discs.get(n), offset=bool(ma.get('offset', 0))))
        if args.agg == 'median' and len(preds) > 1:
            ux = float(np.median([p[0] for p in preds]))
            uy = float(np.median([p[1] for p in preds]))
        else:
            ux = float(np.mean([p[0] for p in preds]))
            uy = float(np.mean([p[1] for p in preds]))
        rows.append((n, ux * W, uy * H))
        print(f'{n}: ({ux*W:.1f}, {uy*H:.1f})')
    df = pd.DataFrame(rows, columns=['id', 'x', 'y'])
    if args.save_raw:
        df.to_csv(args.save_raw, index=False)
    if args.split == 'test':
        sub = pd.read_csv(f'{DATA}/sample_submission.csv')
        val = {}
        for _, r in df.iterrows():
            key = str(int(r.id))   # sample_submission uses ids without leading zeros
            val[f'{key}_Fovea_X'] = r.x
            val[f'{key}_Fovea_Y'] = r.y
        sub['value'] = sub['ImageID'].map(val)
        assert sub['value'].notna().all(), 'missing preds'
        sub.to_csv(args.out, index=False)
        print('wrote', args.out)
        print(sub.head(6).to_string(index=False))


if __name__ == '__main__':
    main()
