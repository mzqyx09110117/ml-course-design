"""Per-axis CV evaluation of baseline vs disc-channel models."""
import sys
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
import numpy as np, torch, pandas as pd
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images
from predict import load_model, predict_one

torch.set_num_threads(8)
gt = load_gt(); tr = sorted(gt); imgs, _, sizes = load_all_images(1024)
dc = pd.read_csv('disc_centers.csv', dtype={'id': str})
disc = {r.id: (float(r.dx), float(r.dy)) for r in dc.itertuples()}
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))


def evaluate(run_fmt):
    ex, ey = [], []
    for f in range(5):
        m, a = load_model(run_fmt.format(f) + '/best.pt')
        for i in splits[f][1]:
            n = tr[i]; W, H = sizes[n]
            ux, uy = predict_one(m, imgs[n], size=a.get('size', 512), tta_flip=False,
                                 disc=disc.get(n), offset=bool(a.get('offset', 0)))
            ex.append(ux * W - gt[n][0]); ey.append(uy * H - gt[n][1])
    ex, ey = np.array(ex), np.array(ey)
    return np.sqrt(np.mean(ex ** 2)), np.sqrt(np.mean(ey ** 2)), np.sqrt((np.mean(ex**2)+np.mean(ey**2))/2)


for name, pre in [('baseline', 'runs/ce_f{}_60'), ('disc', 'runs/disc_f{}')]:
    rx, ry, r = evaluate(pre)
    print(f'{name:9s} x-RMSE {rx:6.2f}  y-RMSE {ry:6.2f}  overall {r:6.2f}')
