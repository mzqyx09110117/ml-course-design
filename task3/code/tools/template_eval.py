"""Classical fovea template matching, evaluated on CV folds.

For each fold: build a mean fovea template from training patches, match on
validation images at several scales (TM_CCOEFF_NORMED).
"""
import sys
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
import numpy as np, cv2
from sklearn.model_selection import StratifiedKFold
from fovea_lib import DATA, load_gt, load_all_images

gt = load_gt(); tr = sorted(gt)
imgs, _, sizes = load_all_images(1024)
grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, grp))
PATCH = 160


def prep(img1024):
    g = cv2.cvtColor(img1024, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 60)
    x = g / (bg + 1e-3)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.astype(np.float32)


def patch_at(x, cx, cy, size=PATCH):
    h = size // 2
    x0 = int(round(cx - h)); y0 = int(round(cy - h))
    x0 = max(0, min(x.shape[1] - size, x0)); y0 = max(0, min(x.shape[0] - size, y0))
    p = x[y0:y0 + size, x0:x0 + size]
    return (p - p.mean()) / (p.std() + 1e-6)


def build_template(names):
    ps = []
    for n in names:
        W, H = sizes[n]
        x = prep(imgs[n])
        ps.append(patch_at(x, gt[n][0] / W * 1024, gt[n][1] / H * 1024))
    return np.mean(ps, 0)


def match(x, tmpl, scales=(0.8, 0.9, 1.0, 1.1, 1.25)):
    best = (-2, None)
    for sc in scales:
        t = cv2.resize(tmpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= x.shape[0] or t.shape[1] >= x.shape[1]:
            continue
        r = cv2.matchTemplate(x, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            cx = loc[0] + t.shape[1] / 2; cy = loc[1] + t.shape[0] / 2
            best = (mx, (cx, cy))
    return best


errs = []
for f in range(5):
    trn = [tr[i] for i in splits[f][0]]
    val = [tr[i] for i in splits[f][1]]
    tmpl = build_template(trn)
    for n in val:
        W, H = sizes[n]
        x = prep(imgs[n])
        score, (cx, cy) = match(x, tmpl)
        from fovea_lib import heatmap_peak  # noqa
        print(f'{n} score {score:.2f} pred ({cx*W/1024:.0f},{cy*H/1024:.0f}) gt ({gt[n][0]:.0f},{gt[n][1]:.0f})')
        errs.append((cx * W / 1024 - gt[n][0], cy * H / 1024 - gt[n][1]))
e = np.array(errs)
print('template RMSE %.1f median %.1f' % (np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2),
                                          np.median(np.hypot(e[:, 0], e[:, 1]))))
np.save('tmpl_errs.npy', e)
