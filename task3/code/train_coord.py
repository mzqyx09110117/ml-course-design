"""Direct coordinate regression baseline (different inductive bias)."""
import argparse, os, time, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision
from fovea_lib import FoveaDataset, load_all_images, load_gt


class CoordNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, freeze_stem=True):
        super().__init__()
        m = getattr(torchvision.models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.frozen = [self.stem, self.l1] if freeze_stem else []
        if freeze_stem:
            for p in self.stem.parameters(): p.requires_grad_(False)
            for p in self.l1.parameters(): p.requires_grad_(False)
        ch = 512 if backbone in ('resnet18', 'resnet34') else 2048
        self.head = nn.Sequential(nn.Linear(ch, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(256, 2))

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen: m.eval()
        return self

    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def run(args):
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    outdir = f'/home/mzqyx09110117/桌面/test3/fovea_work/runs/{args.name}'
    os.makedirs(outdir, exist_ok=True)
    imgs, names, sizes = load_all_images(1024)
    gt = load_gt(); train_names = sorted(gt)
    grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in train_names])
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    splits = list(skf.split(train_names, grp))
    tr = [train_names[i] for i in splits[args.fold][0]]
    va = [train_names[i] for i in splits[args.fold][1]]
    ds_tr = FoveaDataset(tr, imgs, sizes, gt, 512, 1024, 128, train=True, sigma=1.2)
    ds_va = FoveaDataset(va, imgs, sizes, gt, 512, 1024, 128, train=False, sigma=1.2)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=4, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=2)
    model = CoordNet(args.backbone)
    opt = torch.optim.AdamW([
        {'params': [p for p in model.l2.parameters()] + [p for p in model.l3.parameters()] +
                   [p for p in model.l4.parameters()], 'lr': 2e-4},
        {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    spe = len(dl_tr); total = args.epochs * spe; warm = 3 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best = 1e9; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, hm, uv in dl_tr:
            opt.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(x))
            loss = F.l1_loss(pred, uv) * 100.0
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            model.eval(); es = []
            with torch.no_grad():
                for x, hm, uv in dl_va:
                    pred = torch.sigmoid(model(x))
                    es.append((pred - uv).cpu().numpy())
            e = np.concatenate(es); rm = float(np.sqrt((np.mean((e[:, 0] * 2992) ** 2) +
                                                        np.mean((e[:, 1] * 2000) ** 2)) / 2))
            print(f'ep {ep+1}/{args.epochs} loss {tot/len(dl_tr):.3f} valRMSE~{rm:.1f}px [{time.time()-t0:.0f}s]', flush=True)
            if rm < best:
                best = rm
                torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': rm}, f'{outdir}/best.pt')
    torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': best}, f'{outdir}/last.pt')
    print('DONE', best, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='coord_f0'); ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--threads', type=int, default=8)
    run(ap.parse_args())
