import numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, normalize, heatmap_peak
from predict import load_model
torch.set_num_threads(8)
gt=load_gt(); tr=sorted(gt); imgs,_,sizes=load_all_images(1024)
grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))
runs=['runs/ce_f0_60','runs/ce_f1_60','runs/ce_f2_60','runs/ce_f3_60','runs/ce_f4_60']

def pred(model,img1024,size,variants):
    outs=[]
    for b,g in variants:
        im=cv2.resize(img1024,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        im=np.clip((im**g)*b,0,1)
        x=torch.from_numpy(normalize(im)).unsqueeze(0)
        with torch.no_grad():
            o=model(x); p=torch.softmax(o.flatten(1),1).view_as(o)
            px,py=heatmap_peak(p); hm=o.shape[-1]
        outs.append(((px+0.5)/hm,(py+0.5)/hm))
    a=np.array(outs); return np.mean(a,0)

varsets={'none':[(1.0,1.0)], 'bright':[(0.95,1.0),(1.0,1.0),(1.05,1.0)],
         'gamma':[(1.0,0.9),(1.0,1.0),(1.0,1.12)], 'both':[(0.95,0.95),(1.0,1.0),(1.05,1.08)]}
for name,vs in varsets.items():
    es=[]
    for f,run in enumerate(runs):
        m,a=load_model(run+'/best.pt')
        for i in splits[f][1]:
            n=tr[i]; W,H=sizes[n]
            ux,uy=pred(m,imgs[n],512,vs)
            es.append((ux*W-gt[n][0],uy*H-gt[n][1]))
    e=np.array(es)
    print(f'{name:8s} RMSE {np.sqrt((np.mean(e[:,0]**2)+np.mean(e[:,1]**2))/2):.2f} '
          f'median {np.median(np.hypot(e[:,0],e[:,1])):.1f} max {np.hypot(e[:,0],e[:,1]).max():.0f}', flush=True)
