import numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images
from stage2_lib import load_native_images, stage2_predict
from predict import load_model, predict_one
torch.set_num_threads(4)
gt=load_gt(); tr=sorted(gt.keys()); imgs,_,sizes=load_all_images(1024); nat,_=load_native_images(set(tr))
grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
va=[tr[i] for i in list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))[0][1]]
m1,_=load_model('runs/ce_f0_60/best.pt'); m2,a2=load_model('runs/s2_f0/best.pt')
P1=[];P2=[]
for n in va:
    im=cv2.resize(imgs[n],(512,512),interpolation=cv2.INTER_AREA)
    ux,uy=predict_one(m1,im,tta_flip=True)
    W,H=sizes[n]; x1,y1=ux*W,uy*H
    x2,y2=stage2_predict(m2,nat[n],x1,y1,S=a2['S'],tta_flip=True,hm=a2['S']//4)
    P1.append((x1,y1)); P2.append((x2,y2))
P1=np.array(P1);P2=np.array(P2); G=np.array([gt[n] for n in va])
for w in [0,0.1,0.2,0.3,0.4,0.5,0.7,1.0]:
    P=P1*(1-w)+P2*w; e=P-G
    rm=np.sqrt((np.mean(e[:,0]**2)+np.mean(e[:,1]**2))/2)
    print(f'w2={w:.1f} RMSE {rm:.2f} median {np.median(np.hypot(e[:,0],e[:,1])):.1f}')
