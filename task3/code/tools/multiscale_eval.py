import numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images
from predict import load_model, predict_one
torch.set_num_threads(6)
gt=load_gt(); tr=sorted(gt); imgs,_,sizes=load_all_images(1024)
grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))
runs=['runs/ce_f0_60','runs/ce_f1_60','runs/ce_f2_60','runs/ce_f3_60','runs/ce_f4_60']
P={ '512':[], '384':[], '448':[], '256':[] }
G=[]; T=[]
for f,run in enumerate(runs):
    m,_=load_model(run+'/best.pt')
    for i in splits[f][1]:
        n=tr[i]; W,H=sizes[n]
        for s in P:
            ux,uy=predict_one(m,imgs[n],size=int(s),tta_flip=False)
            P[s].append((ux*W,uy*H))
        G.append((gt[n][0],gt[n][1])); T.append(n)
for s in P: P[s]=np.array(P[s])
G=np.array(G)
def rm(a,b): 
    e=a-b; return np.sqrt((np.mean(e[:,0]**2)+np.mean(e[:,1]**2))/2), np.median(np.hypot(e[:,0],e[:,1])), np.hypot(e[:,0],e[:,1]).max()
for s in P: print(s, 'RMSE %.2f med %.1f max %.0f'%rm(P[s],G))
for combo in [('512','384'),('512','448'),('512','384','448'),('512','384','448','256')]:
    a=np.mean([P[s] for s in combo],0)
    print('+'.join(combo),'RMSE %.2f med %.1f max %.0f'%rm(a,G))
