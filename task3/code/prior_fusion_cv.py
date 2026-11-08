"""Bayesian soft-prior fusion on the heatmap, evaluated on CV."""
import sys
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
import numpy as np, cv2, torch, pandas as pd
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, normalize, heatmap_peak, disc_channel
from predict import load_model

torch.set_num_threads(8)
gt = load_gt(); tr = sorted(gt); imgs, _, sizes = load_all_images(1024)
dc = pd.read_csv('disc_centers.csv', dtype={'id': str})
disc = {r.id: (float(r.dx), float(r.dy)) for r in dc.itertuples()}
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))
runs = ['runs/ce_f0_60', 'runs/ce_f1_60', 'runs/ce_f2_60', 'runs/ce_f3_60', 'runs/ce_f4_60']

# training offset stats
absdx, dy = [], []
for n, (fx, fy) in gt.items():
    W, H = sizes[n]; dd = disc[n]
    absdx.append(abs(fx / W - dd[0])); dy.append(fy / H - dd[1])
absdx, dy = np.array(absdx), np.array(dy)
mx, sx = absdx.mean(), absdx.std()
my, sy = dy.mean(), dy.std()
print('prior |dx| %.3f±%.3f  dy %.3f±%.3f' % (mx, sx, my, sy))


def predict_prior(model, n, kx, ky):
    W, H = sizes[n]
    s = 512
    im = cv2.resize(imgs[n], (s, s), interpolation=cv2.INTER_AREA)
    arr = normalize(im.astype(np.float32) / 255.0)
    if model.stem[0].in_channels == 4:
        dd = disc[n]
        arr = np.concatenate([arr, disc_channel(dd[0] * s, dd[1] * s, s)[None]], 0)
    with torch.no_grad():
        o = model(torch.from_numpy(arr).unsqueeze(0))
        p = torch.softmax(o.flatten(1), 1).view_as(o)[0, 0]
    hm = p.shape[-1]
    xs = (torch.arange(hm, dtype=torch.float32) + 0.5) / hm
    ys = (torch.arange(hm, dtype=torch.float32) + 0.5) / hm
    dxg = xs[None, :] - disc[n][0]
    dyg = ys[:, None] - disc[n][1]
    prior = torch.exp(-((dxg.abs() - mx) ** 2) / (2 * (kx * sx) ** 2)) * \
            torch.exp(-((dyg - my) ** 2) / (2 * (ky * sy) ** 2))
    q = p * prior
    if float(q.sum()) < 1e-9:
        q = p
    px, py = heatmap_peak(q[None, None])
    return (px + 0.5) / hm, (py + 0.5) / hm


for kx, ky in [(1, 1), (1.5, 1.5), (2, 2), (3, 3), (2, 3), (3, 2), (10, 10)]:
    es = []
    for f, run in enumerate(runs):
        m, a = load_model(run + '/best.pt')
        for i in splits[f][1]:
            n = tr[i]; W, H = sizes[n]
            ux, uy = predict_prior(m, n, kx, ky)
            es.append((ux * W - gt[n][0], uy * H - gt[n][1]))
    e = np.array(es)
    rm = np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)
    print(f'kx={kx} ky={ky}: CV RMSE {rm:.2f} median {np.median(np.hypot(e[:,0],e[:,1])):.1f} '
          f'x-RMSE {np.sqrt(np.mean(e[:,0]**2)):.1f} y-RMSE {np.sqrt(np.mean(e[:,1]**2)):.1f}')
