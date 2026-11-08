import argparse, os, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from fovea_lib import FoveaNet, load_gt, heatmap_peak
from stage2_lib import load_native_images, Stage2Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--S', type=int, default=512)
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--lr_enc', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--coord_w', type=float, default=0.01)
    ap.add_argument('--sigma', type=float, default=1.2)
    ap.add_argument('--jitter', type=float, default=64.0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    outdir = f'/home/mzqyx09110117/桌面/test3/fovea_work/runs/{args.name}'
    os.makedirs(outdir, exist_ok=True)
    print(json.dumps(vars(args)), flush=True)

    gt = load_gt()
    train_names = sorted(gt.keys())
    imgs, sizes = load_native_images(set(train_names))
    print('loaded native images', len(imgs), 'RAM ok', flush=True)
    grp = np.array([0 if sizes[n][0] == 1956 else 1 for n in train_names])
    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(train_names, grp))
    tr = [train_names[i] for i in splits[args.fold][0]]
    va = [train_names[i] for i in splits[args.fold][1]]
    print(f'fold {args.fold}: train {len(tr)} val {len(va)}', flush=True)

    hm = args.S // 4
    ds_tr = Stage2Dataset(tr, imgs, sizes, gt, args.S, hm, args.sigma, train=True, jitter=args.jitter)
    ds_va = Stage2Dataset(va, imgs, sizes, gt, args.S, hm, args.sigma, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=1,
                       persistent_workers=True)

    model = FoveaNet(args.backbone, pretrained=True, freeze_stem=True, width=args.width)
    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if ('lat' in n or 'smooth' in n or 'head' in n) else enc_params).append(p)
    opt = torch.optim.AdamW([{'params': enc_params, 'lr': args.lr_enc},
                             {'params': head_params, 'lr': args.lr_head}], weight_decay=args.wd)
    spe = len(dl_tr); total = args.epochs * spe; warm = 3 * spe

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def evaluate():
        model.eval()
        errs = []; base = 0
        with torch.no_grad():
            for x, target, uv in dl_va:
                o = model(x)
                probs = torch.softmax(o.flatten(1), dim=1).view_as(o)
                for b in range(x.shape[0]):
                    px, py = heatmap_peak(probs[b:b + 1])
                    ux, uy = (px + 0.5) / hm, (py + 0.5) / hm
                    errs.append((float(ux * args.S - 0.5 - args.S / 2),
                                 float(uy * args.S - 0.5 - args.S / 2)))
                base += x.shape[0]
        model.train()
        e = np.array(errs)
        return float(np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)), errs

    best = 1e9; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = tot_ce = tot_co = 0.0
        for x, target, uv in dl_tr:
            opt.zero_grad(set_to_none=True)
            o = model(x)
            q = target / target.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            logp = F.log_softmax(o.flatten(1), dim=1).view_as(o)
            ce = -(q * logp).sum(dim=(1, 2, 3)).mean()
            cx = uv[:, 0] * hm - 0.5; cy = uv[:, 1] * hm - 0.5
            p = logp.exp()
            ys = torch.arange(hm, device=o.device, dtype=o.dtype).view(1, 1, hm, 1)
            xs = torch.arange(hm, device=o.device, dtype=o.dtype).view(1, 1, 1, hm)
            px = (p * xs).sum(dim=(1, 2, 3)); py = (p * ys).sum(dim=(1, 2, 3))
            co = (px - cx).abs().mean() + (py - cy).abs().mean()
            loss = ce + args.coord_w * co
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); tot_ce += ce.item(); tot_co += co.item()
        n = len(dl_tr)
        line = f'ep {ep+1:3d}/{args.epochs} loss {tot/n:.4f} ce {tot_ce/n:.4f} co {tot_co/n:.2f}'
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            rmse, errs = evaluate()
            line += f' | val(gt-centred) {rmse:.2f}px'
            if rmse < best:
                best = rmse
                torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': rmse},
                           f'{outdir}/best.pt')
                line += ' *'
        print(f'{line} [{time.time()-t0:.0f}s]', flush=True)
    torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': best}, f'{outdir}/last.pt')
    print(f'DONE stage2 fold {args.fold} best {best:.2f} time {(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
