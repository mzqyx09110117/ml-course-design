import numpy as np, cv2, torch, pandas as pd
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, DATA
from predict import load_model, predict_one
torch.set_num_threads(4)
gt=load_gt(); tr=sorted(gt); imgs,_,sizes=load_all_images(1024)
grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))
runs=['runs/ce_f0_60','runs/ce_f1_60','runs/ce_f2_60','runs/ce_f3_60','runs/ce_f4_60']

def disc_center(n):
    im=cv2.resize(imgs[n],(1024,1024))
    g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    g=cv2.GaussianBlur(g,(0,0),7)
    m=(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)>12)
    thr=np.percentile(g[m],99.0)
    bw=(g>thr).astype(np.uint8)
    nc,lab,st,cent=cv2.connectedComponentsWithStats(bw,8)
    if nc<2: return np.nan,np.nan
    idx=1+np.argmax(st[1:,4]); return cent[idx]/1024

rows=[]
for f,run in enumerate(runs):
    m,_=load_model(run+'/best.pt')
    for i in splits[f][1]:
        n=tr[i]
        im=cv2.resize(imgs[n],(512,512),interpolation=cv2.INTER_AREA)
        ux,uy=predict_one(m,im,tta_flip=False)
        W,H=sizes[n]; dx,dy=disc_center(n)
        rows.append(dict(n=n,pred_x=ux*W,pred_y=uy*H,gt_x=gt[n][0],gt_y=gt[n][1],W=W,H=H,disc_x=dx,disc_y=dy))
df=pd.DataFrame(rows)
df['ex']=df.pred_x-df.gt_x; df['ey']=df.pred_y-df.gt_y
df['err']=np.hypot(df.ex,df.ey)
df['gdx']=(df.gt_x-df.disc_x*df.W)/df.W
df['gdy']=(df.gt_y-df.disc_y*df.H)/df.H
df['pdx']=(df.pred_x-df.disc_x*df.W)/df.W
df['pdy']=(df.pred_y-df.disc_y*df.H)/df.H
df.to_csv('cv_preds.csv',index=False)
print(df.sort_values('err',ascending=False).head(12)[['n','ex','ey','err','gdx','gdy','pdx','pdy']].to_string(index=False))
print('\nGT dx range:',df.gdx.min().round(3),df.gdx.max().round(3),'| GT dy range:',df.gdy.min().round(3),df.gdy.max().round(3))
print('Outliers (err>50):')
o=df[df.err>50]
print(o[['n','ex','ey','err','gdx','gdy','pdx','pdy']].to_string(index=False))
print('\nRMSE',np.sqrt((np.mean(df.ex**2)+np.mean(df.ey**2))/2),'median',df.err.median())
