import torch, torchvision, time
import torch.nn as nn
torch.set_num_threads(16)
def bench(name, model, size=512, bs=8, iters=3):
    model.train()
    x=torch.randn(bs,3,size,size); y=torch.randn(bs,1,size//4,size//4)
    opt=torch.optim.AdamW(model.parameters(),1e-3)
    # warmup
    for _ in range(1):
        opt.zero_grad(); o=model(x); loss=o.float().pow(2).mean(); loss.backward(); opt.step()
    t=time.time()
    for _ in range(iters):
        opt.zero_grad(); o=model(x); loss=o.float().pow(2).mean(); loss.backward(); opt.step()
    dt=(time.time()-t)/iters
    print(f'{name} size={size} bs={bs}: {dt:.2f}s/iter, {dt/bs*1000:.0f} ms/img')
for s in [384,512,640]:
    m=torchvision.models.resnet18(weights=None)
    bench('resnet18-all', m, s)
m=torchvision.models.resnet18(weights=None)
for p in list(m.conv1.parameters())+list(m.layer1.parameters()): p.requires_grad_(False)
bench('resnet18-frozen-stem', m, 512)
import timm
m=timm.create_model('mobilenetv3_large_100',pretrained=False,num_classes=0)
bench('mbv3', m, 512)
m=timm.create_model('efficientnet_b0',pretrained=False,num_classes=0)
bench('effb0', m, 512)
