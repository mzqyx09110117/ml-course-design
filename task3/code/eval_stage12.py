import argparse, os
import numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, heatmap_peak
from stage2_lib import load_native_images, stage2_predict
from predict import load_model, predict_one

ap = argparse.ArgumentParser()
ap.add_argument('--s1', required=True)
ap.add_argument('--s2', required=True)
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--no_flip', action='store_true')
ap.add_argument('--s2_scales', default='1.0')
args = ap.parse_args()
torch.set_num_threads(12)

gt = load_gt()
tr = sorted(gt.keys())
imgs1024, _, sizes = load_all_images(1024)
imgs_nat, _ = load_native_images(set(tr))
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))
va = [tr[i] for i in splits[args.fold][1]]

m1, a1 = load_model(args.s1)
m2, a2 = load_model(args.s2)
scales = [float(s) for s in args.s2_scales.split(',')]

e1all, e2all = [], []
for n in va:
    im512 = cv2.resize(imgs1024[n], (512, 512), interpolation=cv2.INTER_AREA)
    ux, uy = predict_one(m1, im512, tta_flip=not args.no_flip)
    W, H = sizes[n]
    x1, y1 = ux * W, uy * H
    # stage-2 refinement (flip TTA inside, multi-scale outside)
    ps = [stage2_predict(m2, imgs_nat[n], x1, y1, S=a2['S'], tta_flip=not args.no_flip,
                         hm=a2['S'] // 4, scale=s) for s in scales]
    x2 = float(np.mean([p[0] for p in ps])); y2 = float(np.mean([p[1] for p in ps]))
    gx, gy = gt[n]
    e1all.append((x1 - gx, y1 - gy)); e2all.append((x2 - gx, y2 - gy))
    print(f'{n} gt=({gx:.0f},{gy:.0f}) s1=({x1:.0f},{y1:.0f}) err1={np.hypot(x1-gx,y1-gy):.0f} '
          f's2=({x2:.0f},{y2:.0f}) err2={np.hypot(x2-gx,y2-gy):.0f}')

for tag, e in [('stage1', np.array(e1all)), ('stage1+2', np.array(e2all))]:
    rmse = np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)
    print(f'{tag}: RMSE {rmse:.2f}px median {np.median(np.hypot(e[:,0],e[:,1])):.1f} max {np.hypot(e[:,0],e[:,1]).max():.0f}')
