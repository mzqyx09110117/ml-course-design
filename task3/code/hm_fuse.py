import sys, glob, os
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
import numpy as np, cv2, torch, pandas as pd
import torch.nn.functional as F
from fovea_lib import DATA, load_all_images, normalize, heatmap_peak, disc_channel
from predict import load_model

torch.set_num_threads(8)
ckpts = sys.argv[1].split(',')
out = sys.argv[2]
scales = [float(s) for s in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['1.0'])]
models = [load_model(c) for c in ckpts]
d = pd.read_csv('/home/mzqyx09110117/桌面/test3/fovea_work/disc_centers.csv', dtype={'id': str})
disc = {r.id: (float(r.dx), float(r.dy)) for r in d.itertuples()}
imgs, names, sizes = load_all_images(1024)

rows = []
for f in sorted(glob.glob(f'{DATA}/test/*.jpg')):
    n = os.path.basename(f).split('.')[0]
    W, H = sizes[n]
    hms = []
    for m, a in models:
        for sc in scales:
            base = a.get('size', 512)
            s = max(64, int(round(base * sc / 32)) * 32)
            im = cv2.resize(imgs[n], (s, s), interpolation=cv2.INTER_AREA)
            arr = normalize(im / 255.0)
            if m.stem[0].in_channels == 4:
                dd = disc.get(n, (0.5, 0.5))
                arr = np.concatenate([arr, disc_channel(dd[0] * s, dd[1] * s, s,
                                                         sigma=10.0 * s / 512.0)[None]], 0)
            x = torch.from_numpy(arr).unsqueeze(0)
            with torch.no_grad():
                o = m(x)
                p = torch.softmax(o.flatten(1), 1).view_as(o)
                if a.get('offset', 0):
                    continue  # offset models not supported here
                hms.append(F.interpolate(p, size=(128, 128), mode='bilinear', align_corners=False))
    hm = torch.stack(hms).mean(0)
    px, py = heatmap_peak(hm)
    ux, uy = (px + 0.5) / 128, (py + 0.5) / 128
    rows.append((n, ux * W, uy * H))
    print(n, round(ux * W, 1), round(uy * H, 1))
pd.DataFrame(rows, columns=['id', 'x', 'y']).to_csv(out, index=False)
