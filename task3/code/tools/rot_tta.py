import numpy as np, cv2, torch, sys
from sklearn.model_selection import StratifiedKFold
from fovea_lib import load_gt, load_all_images, normalize, heatmap_peak
from predict import load_model
torch.set_num_threads(4)
gt=load_gt(); tr=sorted(gt); imgs,_,sizes=load_all_images(1024)
grp=np.array([0 if sizes[n][0]==1956 else 1 for n in tr])
va=[tr[i] for i in list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,grp))[1][1]]
m,_=load_model('runs/ce_f1_60/best.pt')
S=512
def predict_rot(model,img,angles):
    outs=[]
    for a in angles:
        M=cv2.getRotationMatrix2D((S/2,S/2),a,1.0)
        im=cv2.warpAffine(img,M,(S,S),borderValue=0)
        x=torch.from_numpy(normalize(im.astype(np.float32)/255)).unsqueeze(0)
        with torch.no_grad():
            o=model(x); p=torch.softmax(o.flatten(1),1).view_as(o)
            px,py=heatmap_peak(p); hm=o.shape[-1]
        po=np.array([(px+0.5)/hm*S-0.5,(py+0.5)/hm*S-0.5,1])
        pi=np.linalg.solve(M[:,:2],po[:2]-M[:,2])
        outs.append(((pi[0]+0.5)/S,(pi[1]+0.5)/S))
    return np.mean(outs,0)
for name,angs in [('none',[0]),('rot+-5',[-5,0,5]),('rot+-8',[-8,0,8]),('rot4',[-8,-4,4,8])]:
    es=[]
    for n in va:
        im=cv2.resize(imgs[n],(S,S),interpolation=cv2.INTER_AREA)
        ux,uy=predict_rot(m,im,angs)
        W,H=sizes[n]; es.append((ux*W-gt[n][0],uy*H-gt[n][1]))
    e=np.array(es); print(f'{name:8s} RMSE {np.sqrt((np.mean(e[:,0]**2)+np.mean(e[:,1]**2))/2):.2f} median {np.median(np.hypot(e[:,0],e[:,1])):.1f} max {np.hypot(e[:,0],e[:,1]).max():.0f}')
