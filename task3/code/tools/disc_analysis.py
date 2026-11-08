import numpy as np, cv2, pandas as pd
from fovea_lib import DATA, load_gt
gt=load_gt()
res=[]
for n,(fx,fy) in sorted(gt.items()):
    im=cv2.imread(f'{DATA}/train/{n}.jpg'); H,W=im.shape[:2]
    im=cv2.resize(im,(1024,1024))
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    g=cv2.GaussianBlur(g,(0,0),7)
    # fundus mask: non-black
    mask=(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)>12).astype(np.uint8)
    # brightest blob = disc
    thr=np.percentile(g[mask>0],99.0)
    bw=(g>thr).astype(np.uint8)
    ncomp,lab,stats,cent=cv2.connectedComponentsWithStats(bw,8)
    if ncomp<2: continue
    idx=1+np.argmax(stats[1:,4]); dx,dy=cent[idx]
    # disc center robust: use centroid of bright region within radius
    # fovea normalized
    ux,uy=fx/W,fy/H
    res.append((n,ux,uy,dx/1024,dy/1024,ux-dx/1024,uy-dy/1024))
a=np.array([r[5:] for r in res])
print('n',len(res))
print('dx mean %.3f std %.3f | dy mean %.3f std %.3f'%(a[:,0].mean(),a[:,0].std(),a[:,1].mean(),a[:,1].std()))
print('|dx| range', np.abs(a[:,0]).min(), np.abs(a[:,0]).max())
print('abs dx percentiles', np.percentile(np.abs(a[:,0]),[10,50,90,100]))
# baseline: predict fovea = disc + mean offset
pred=np.array([[r[3]+a[:,0].mean(), r[4]+a[:,1].mean()] for r in res])
err=np.array([[ (pred[i,0]-res[i][1]), (pred[i,1]-res[i][2]) ] for i in range(len(res))])
print('baseline RMSE norm', np.sqrt((np.mean(err[:,0]**2)+np.mean(err[:,1]**2))/2))
print('baseline RMSE px ~', np.sqrt((np.mean(err[:,0]**2)+np.mean(err[:,1]**2))/2)*2992)
print('sign of dx counts:', (a[:,0]>0).sum(), (a[:,0]<0).sum())
for r in res[:12]: print(r[0], 'dx %.3f dy %.3f'%(r[5],r[6]))
