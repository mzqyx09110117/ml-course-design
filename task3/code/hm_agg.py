import numpy as np, cv2, torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, normalize, heatmap_peak
from predict import load_model
torch.set_num_threads(8)
gt=load_gt(); tr=sorted(gt); imgs,_,sizes=load_all_images(1024)
grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))
runs=['runs/ce_f0_60','runs/ce_f1_60','runs/ce_f2_60','runs/ce_f3_60','runs/ce_f4_60']

def heatmaps(model,img1024,sizes_list):
    hms=[]
    for s in sizes_list:
        im=cv2.resize(img1024,(s,s),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(normalize(im.astype(np.float32)/255)).unsqueeze(0)
        with torch.no_grad():
            o=model(x); p=torch.softmax(o.flatten(1),1).view_as(o)
        hms.append(p)
    return hms

def peak_of(h):
    px,py=heatmap_peak(h); hm=h.shape[-1]
    return (px+0.5)/hm,(py+0.5)/hm

def ev(tag, vals):
    e=np.array(vals)
    print(f'{tag:22s} RMSE {np.sqrt((np.mean(e[:,0]**2)+np.mean(e[:,1]**2))/2):.2f} '
          f'median {np.median(np.hypot(e[:,0],e[:,1])):.1f} max {np.hypot(e[:,0],e[:,1]).max():.0f}', flush=True)

for tag, mode, sizes_list in [('coord 512','coord',[512]),('coord 512+384','coord',[512,384]),
                              ('hm 512','hm',[512]),('hm 512+384','hm',[512,384])]:
    vals=[]
    for f,run in enumerate(runs):
        m,a=load_model(run+'/best.pt')
        for i in splits[f][1]:
            n=tr[i]; W,H=sizes[n]
            hms=heatmaps(m,imgs[n],sizes_list)
            if mode=='coord':
                ps=[peak_of(h) for h in hms]; p=np.mean(ps,0)
            else:
                tgt=hms[0].shape[-1]
                ups=[F.interpolate(h,size=(tgt,tgt),mode='bilinear',align_corners=False) for h in hms]
                p=peak_of(torch.stack(ups).mean(0))
            vals.append((p[0]*W-gt[n][0],p[1]*H-gt[n][1]))
    ev(tag, vals)
