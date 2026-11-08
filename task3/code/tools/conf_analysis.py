import sys
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
import numpy as np, cv2, torch, pandas as pd
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, normalize, heatmap_peak
from predict import load_model

torch.set_num_threads(4)
gt = load_gt(); tr = sorted(gt); imgs, _, sizes = load_all_images(1024)
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))
runs = ['runs/ce_f0_60', 'runs/ce_f1_60', 'runs/ce_f2_60', 'runs/ce_f3_60', 'runs/ce_f4_60']

def peak_prob(model, img1024, size=512):
    im = cv2.resize(img1024, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(normalize(im.astype(np.float32) / 255.0)).unsqueeze(0)
    with torch.no_grad():
        o = model(x); p = torch.softmax(o.flatten(1), 1).view_as(o)
    return float(p.max()), heatmap_peak(p)

rows = []
for f, run in enumerate(runs):
    m, _ = load_model(run + '/best.pt')
    for i in splits[f][1]:
        n = tr[i]; W, H = sizes[n]
        conf, (px, py) = peak_prob(m, imgs[n])
        ux, uy = (px + .5) / 128, (py + .5) / 128
        rows.append((n, ux * W, uy * H, conf, gt[n][0], gt[n][1]))
df = pd.DataFrame(rows, columns=['n', 'px', 'py', 'conf', 'gx', 'gy'])
df['err'] = np.hypot(df.px - df.gx, df.py - df.gy)
df['ex'] = df.px - df.gx; df['ey'] = df.py - df.gy
print('CV conf vs error: corr %.2f' % np.corrcoef(df.conf, df.err)[0, 1])
print(df[['conf', 'err']].describe().round(3).to_string())
df.to_csv('cv_conf.csv', index=False)
# population prior
mx, my = df.gx.mean(), df.gy.mean()
print('prior mean', round(mx), round(my))
for th in [0.06, 0.08, 0.10, 0.12, 0.15]:
    low = df.conf < th
    print(f'conf<{th}: n={low.sum()} median err {df.err[low].median():.1f} vs high {df.err[~low].median():.1f}')
