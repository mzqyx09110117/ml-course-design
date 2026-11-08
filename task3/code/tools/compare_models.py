import numpy as np, cv2, torch, glob, os
from fovea_lib import DATA, load_all_images
from predict import load_model, predict_one
torch.set_num_threads(8)
imgs,names,sizes=load_all_images(1024)
runs=['runs/ce_f0_60','runs/ce_f1_60','runs/ce_f2_60','runs/ce_f3_60','runs/ce_f4_60','runs/full_r18_s7']
models=[load_model(r+'/best.pt')[0] for r in runs]
files=sorted(glob.glob(f'{DATA}/test/*.jpg'))
print('img   5fold_med      full_r18     diff')
for f in files:
    n=os.path.basename(f).split('.')[0]; W,H=sizes[n]
    im=cv2.resize(imgs[n],(512,512),interpolation=cv2.INTER_AREA)
    ps=[predict_one(m,im,tta_flip=False) for m in models]
    xs=np.array([p[0]*W for p in ps]); ys=np.array([p[1]*H for p in ps])
    mx,my=np.median(xs[:5]),np.median(ys[:5])
    print(f'{n} ({mx:.0f},{my:.0f})  ({xs[5]:.0f},{ys[5]:.0f})  ({xs[5]-mx:+.0f},{ys[5]-my:+.0f})')
