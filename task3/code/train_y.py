"""1D fovea-y regressor with optic-disc channel input."""
import argparse, os, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision
from fovea_lib import FoveaDataset, load_all_images, load_gt


class YNet(nn.Module):
    def __init__(self, backbone='resnet18', in_ch=4):
        super().__init__()
        m = getattr(torchvision.models, backbone)(weights='IMAGENET1K_V1')
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.frozen = [self.stem, self.l1]
        for p in self.stem.parameters(): p.requires_grad_(False)
        for p in self.l1.parameters(): p.requires_grad_(False)
        if in_ch != 3:
            old = self.stem[0]
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding,
                            bias=old.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
                if old.bias is not None and new.bias is not None:
                    new.bias.copy_(old.bias)
            self.stem[0] = new
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(256, 1))

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen: m.eval()
        return self

    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='ynet_f0'); ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = f'/home/mzqyx09110117/桌面/test3/fovea_work/runs/{a.name}'; os.makedirs(out, exist_ok=True)
    imgs, names, sizes = load_all_images(1024); gt = load_gt(); tr = sorted(gt)
    import pandas as pd
    dc = pd.read_csv('/home/mzqyx09110117/桌面/test3/fovea_work/disc_centers.csv', dtype={'id': str})
    disc = {r.id: (float(r.dx), float(r.dy)) for r in dc.itertuples()}
    grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in tr])
    splits = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(tr, grp))
    trn = [tr[i] for i in splits[a.fold][0]]; val = [tr[i] for i in splits[a.fold][1]]
    ds_tr = FoveaDataset(trn, imgs, sizes, gt, 512, 1024, 128, train=True, disc=disc)
    ds_va = FoveaDataset(val, imgs, sizes, gt, 512, 1024, 128, train=False, disc=disc)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=4, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=2)
    model = YNet(); opt = torch.optim.AdamW([
        {'params': list(model.l2.parameters()) + list(model.l3.parameters()) + list(model.l4.parameters()), 'lr': 2e-4},
        {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    spe = len(dl_tr); total = a.epochs * spe; warm = 3 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best = 1e9; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0
        for x, hm, uv in dl_tr:
            opt.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(x))
            loss = F.l1_loss(pred, uv[:, 1]) * 100
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            model.eval(); es = []
            with torch.no_grad():
                for x, hm, uv in dl_va:
                    pred = torch.sigmoid(model(x)); es.append((pred - uv[:, 1]).cpu().numpy())
            e = np.concatenate(es); rm = float(np.sqrt(np.mean((e * 2000) ** 2)))
            print(f'ep {ep+1}/{a.epochs} loss {tot/len(dl_tr):.2f} valY-RMSE {rm:.1f}px [{time.time()-t0:.0f}s]', flush=True)
            if rm < best:
                best = rm
                torch.save({'model': model.state_dict(), 'args': vars(a), 'rmse': rm}, f'{out}/best.pt')
    print('DONE', best, flush=True)


if __name__ == '__main__':
    main()
