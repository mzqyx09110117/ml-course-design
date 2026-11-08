import argparse, os, itertools
import numpy as np, cv2, torch, pandas as pd
from sklearn.model_selection import StratifiedKFold
from fovea_lib import DATA, load_all_images, load_gt
from predict import load_model, predict_one

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True, help='run dirs (one per fold, in fold order)')
ap.add_argument('--no_flip', action='store_true')
ap.add_argument('--scales', default='1.0')
args = ap.parse_args()
torch.set_num_threads(12)

imgs, names, sizes = load_all_images(1024)
gt = load_gt()
tr = sorted(gt.keys())
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))
scales = [float(s) for s in args.scales.split(',')]

all_errs = []
for fold, run in enumerate(args.runs):
    ck = os.path.join(run, 'best.pt')
    model, a = load_model(ck)
    va = [tr[i] for i in splits[fold][1]]
    errs = []
    for n in va:
        im512 = cv2.resize(imgs[n], (512, 512), interpolation=cv2.INTER_AREA)
        ux, uy = predict_one(model, im512, tta_flip=not args.no_flip, scales=scales)
        W, H = sizes[n]
        ex, ey = ux * W - gt[n][0], uy * H - gt[n][1]
        errs.append((n, ex, ey))
        all_errs.append((n, ex, ey))
    e = np.array([(x[1], x[2]) for x in errs])
    rmse = np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)
    print(f'fold {fold} ({run}) {len(va)} imgs RMSE {rmse:.2f}px')
    for n, ex, ey in sorted(errs, key=lambda t: -np.hypot(t[1], t[2]))[:4]:
        print(f'   worst {n}: ({ex:+.0f},{ey:+.0f}) |e|={np.hypot(ex,ey):.0f}')

e = np.array([(x[1], x[2]) for x in all_errs])
rmse = np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)
med = np.median(np.hypot(e[:, 0], e[:, 1]))
print(f'OVERALL {len(e)} imgs: RMSE {rmse:.2f}px  median {med:.1f}px  max {np.hypot(e[:,0],e[:,1]).max():.0f}px')
print(f'  err_x mean {e[:,0].mean():+.1f} std {e[:,0].std():.1f} | err_y mean {e[:,1].mean():+.1f} std {e[:,1].std():.1f}')
