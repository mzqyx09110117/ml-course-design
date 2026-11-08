import sys, numpy as np, torch, pandas as pd
from fovea_lib import *
from predict import load_model, predict_one
ck=sys.argv[1]; fold=int(sys.argv[2]) if len(sys.argv)>2 else 0
model,a=load_model(ck)
imgs,names,sizes=load_all_images(1024); gt=load_gt()
from sklearn.model_selection import StratifiedKFold
tr=sorted(gt.keys()); grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
va=[tr[i] for i in list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))[fold][1]]
errs=[]
for n in va:
    im=cv2.resize(imgs[n],(512,512),interpolation=cv2.INTER_AREA)
    ux,uy=predict_one(model, im, tta_flip=False)
    W,H=sizes[n]; gx,gy=gt[n]
    ex,ey=ux*W-gx, uy*H-gy
    errs.append((n,ex,ey,np.hypot(ex,ey)))
    print(f'{n} pred=({ux*W:.0f},{uy*H:.0f}) gt=({gx:.0f},{gy:.0f}) err=({ex:.0f},{ey:.0f})')
e=np.array([x[3] for x in errs])
print('RMSE', np.sqrt((np.mean([x[1]**2 for x in errs])+np.mean([x[2]**2 for x in errs]))/2), 'median', np.median(e), 'max', e.max())
