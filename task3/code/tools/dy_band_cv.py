"""Evaluate a vertical (dy from optic disc) band prior on CV."""
import sys
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
import numpy as np, cv2, torch, pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, normalize, heatmap_peak, disc_channel
from predict import load_model

torch.set_num_threads(6)
gt = load_gt(); tr = sorted(gt); imgs, _, sizes = load_all_images(1024)
dc = pd.read_csv('disc_centers.csv', dtype={'id': str})
disc = {r.id: (float(r.dx), float(r.dy)) for r in dc.itertuples()}
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))
runs = ['runs/ce_f0_60', 'runs/ce_f1_60', 'runs/ce_f2_60', 'runs/ce_f3_60', 'runs/ce_f4_60']

# training dy stats
dys = []
for n, (fx, fy) in gt.items():
    dd = disc[n]
    dys.append(fy / sizes[n][1] - dd[1])
dys = np.array(dys)
mu, sd = dys.mean(), dys.std()
print('train dy mean %.3f std %.3f' % (mu, sd))


def predict_band(model, n, delta, size=512):
    W, H = sizes[n]
    im = cv2.resize(imgs[n], (size, size), interpolation=cv2.INTER_AREA)
    arr = normalize(im.astype(np.float32) / 255.0)
    if model.stem[0].in_channels == 4:
        dd = disc[n]
        arr = np.concatenate([arr, disc_channel(dd[0] * size, dd[1] * size, size)[None]], 0)
    x = torch.from_numpy(arr).unsqueeze(0)
    with torch.no_grad():
        o = model(x); p = torch.softmax(o.flatten(1), 1).view_as(o)[0, 0]
    hm = p.shape[-1]
    if delta > 0:
        ys = (torch.arange(hm, dtype=torch.float32) + 0.5) / hm
        band = (ys - disc[n][1] - mu).abs() <= delta
        if band.sum() > 0:
            p = p * band.view(hm, 1)
    px, py = heatmap_peak(p[None, None])
    return (px + 0.5) / hm, (py + 0.5) / hm


for delta in [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
    es = []
    for f, run in enumerate(runs):
        m, a = load_model(run + '/best.pt')
        for i in splits[f][1]:
            n = tr[i]; W, H = sizes[n]
            ux, uy = predict_band(m, n, delta)
            es.append((ux * W - gt[n][0], uy * H - gt[n][1]))
    e = np.array(es)
    rm = np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)
    print(f'delta={delta:.2f} CV RMSE {rm:.2f} median {np.median(np.hypot(e[:,0],e[:,1])):.1f} '
          f'y-RMSE {np.sqrt(np.mean(e[:,1]**2)):.1f}')
