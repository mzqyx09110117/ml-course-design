import sys, glob, os
import numpy as np, cv2, pandas as pd, torch
sys.path.insert(0, '/home/mzqyx09110117/桌面/test3/fovea_work')
from fovea_lib import DATA, load_all_images, apply_clahe
from predict import load_model, predict_one

torch.set_num_threads(8)
ckpt, out_raw, mode = sys.argv[1], sys.argv[2], sys.argv[3]  # mode: clahe
imgs, names, sizes = load_all_images(1024)
m, a = load_model(ckpt)
rows = []
for f in sorted(glob.glob(f'{DATA}/test/*.jpg')):
    n = os.path.basename(f).split('.')[0]
    im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    if mode == 'clahe':
        im = apply_clahe(im)
    im1024 = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
    ux, uy = predict_one(m, im1024, size=a.get('size', 512), tta_flip=False, scales=(1.0, 0.75))
    W, H = sizes[n]
    rows.append((n, ux * W, uy * H))
    print(n, round(ux * W, 1), round(uy * H, 1))
pd.DataFrame(rows, columns=['id', 'x', 'y']).to_csv(out_raw, index=False)
