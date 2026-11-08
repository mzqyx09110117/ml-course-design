import argparse, os, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from fovea_lib import (DATA, FoveaNet, FoveaDataset, load_all_images, load_gt,
                       local_softargmax, heatmap_peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='f0')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--inp', type=int, default=1024)
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--lr_enc', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--coord_w', type=float, default=0.01)
    ap.add_argument('--sigma', type=float, default=1.2)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--freeze_stem', type=int, default=1)
    ap.add_argument('--clahe_p', type=float, default=0.0)
    ap.add_argument('--disc_ch', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--haze_p', type=float, default=0.0)
    ap.add_argument('--stride2', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--aug_seed', type=int, default=-1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    np.random.seed(args.aug_seed if args.aug_seed >= 0 else args.seed)
    torch.manual_seed(args.aug_seed if args.aug_seed >= 0 else args.seed)
    outdir = f'/home/mzqyx09110117/桌面/test3/fovea_work/runs/{args.name}'
    os.makedirs(outdir, exist_ok=True)
    print(json.dumps(vars(args)), flush=True)

    imgs, names, sizes = load_all_images(args.inp)
    gt = load_gt()
    train_names = sorted(gt.keys())
    res_group = np.array([0 if sizes[n][0] == 1956 else 1 for n in train_names])
    if args.fold < 0:  # train on all data, no validation
        tr, va = train_names, []
    else:
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        tr_idx, va_idx = list(skf.split(train_names, res_group))[args.fold]
        tr = [train_names[i] for i in tr_idx]
        va = [train_names[i] for i in va_idx]
    print(f'fold {args.fold}: train {len(tr)} val {len(va)} val={va}', flush=True)

    hm = args.size // (2 if args.stride2 else 4)
    disc = None
    if args.disc_ch:
        import pandas as _pd
        _d = _pd.read_csv('/home/mzqyx09110117/桌面/test3/fovea_work/disc_centers.csv', dtype={'id': str})
        disc = {r.id: (float(r.dx), float(r.dy)) for r in _d.itertuples()}
    ds_tr = FoveaDataset(tr, imgs, sizes, gt, args.size, args.inp, hm, train=True, sigma=args.sigma,
                         clahe_p=args.clahe_p, disc=disc, offset_mode=bool(args.offset), haze_p=args.haze_p)
    ds_va = FoveaDataset(va, imgs, sizes, gt, args.size, args.inp, hm, train=False, sigma=args.sigma,
                         disc=disc, offset_mode=bool(args.offset), haze_p=args.haze_p)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=2,
                       persistent_workers=True)

    model = FoveaNet(args.backbone, pretrained=True, freeze_stem=bool(args.freeze_stem), width=args.width,
                     in_ch=4 if args.disc_ch else 3, stride2=bool(args.stride2))
    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if ('lat' in n or 'smooth' in n or 'head' in n) else enc_params).append(p)
    opt = torch.optim.AdamW([{'params': enc_params, 'lr': args.lr_enc},
                             {'params': head_params, 'lr': args.lr_head}], weight_decay=args.wd)
    steps_per_epoch = len(dl_tr)
    total_steps = args.epochs * steps_per_epoch
    warmup = 3 * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def evaluate():
        model.eval()
        errs = []
        base = 0
        with torch.no_grad():
            for x, target, uv in dl_va:
                o = model(x)
                probs = torch.softmax(o.flatten(1), dim=1).view_as(o)
                for b in range(x.shape[0]):
                    n = va[base + b]
                    W, H = sizes[n]
                    gx, gy = gt[n]
                    px, py = heatmap_peak(probs[b:b + 1])
                    # heatmap cell -> normalised (cell centre at (j+0.5)/hm)
                    ux, uy = (px + 0.5) / hm, (py + 0.5) / hm
                    if args.offset and disc is not None:
                        dd = disc[n]
                        errs.append((n, (dd[0] + (ux - 0.5)) * W - gx,
                                     (dd[1] + (uy - 0.5)) * H - gy))
                    else:
                        errs.append((n, ux * W - gx, uy * H - gy))
                base += x.shape[0]
        model.train()
        ex = np.array([e[1] for e in errs]); ey = np.array([e[2] for e in errs])
        rmse = float(np.sqrt((np.mean(ex ** 2) + np.mean(ey ** 2)) / 2))
        return rmse, errs

    best = 1e9
    t0 = time.time()
    hist = []
    for ep in range(args.epochs):
        model.train()
        tot, tot_hm, tot_co = 0.0, 0.0, 0.0
        for x, target, uv in dl_tr:
            opt.zero_grad(set_to_none=True)
            o = model(x)
            # spatial softmax classification with Gaussian soft labels
            q = target / target.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            logp = F.log_softmax(o.flatten(1), dim=1).view_as(o)
            hm_loss = -(q * logp).sum(dim=(1, 2, 3)).mean()
            cx = uv[:, 0] * hm - 0.5
            cy = uv[:, 1] * hm - 0.5
            p = logp.exp()
            ys = torch.arange(hm, device=o.device, dtype=o.dtype).view(1, 1, hm, 1)
            xs = torch.arange(hm, device=o.device, dtype=o.dtype).view(1, 1, 1, hm)
            px = (p * xs).sum(dim=(1, 2, 3))
            py = (p * ys).sum(dim=(1, 2, 3))
            coord_loss = (px - cx).abs().mean() + (py - cy).abs().mean()
            loss = hm_loss + args.coord_w * coord_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); tot_hm += hm_loss.item(); tot_co += coord_loss.item()
        n = len(dl_tr)
        line = f'ep {ep+1:3d}/{args.epochs} loss {tot/n:.4f} hm {tot_hm/n:.5f} co {tot_co/n:.2f}'
        if va and ((ep + 1) % 5 == 0 or ep == args.epochs - 1):
            rmse, errs = evaluate()
            line += f' | val RMSE {rmse:.1f}px'
            if rmse < best:
                best = rmse
                torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': rmse},
                           f'{outdir}/best.pt')
                line += ' *'
        hist.append((ep + 1, tot / n, tot_hm / n, tot_co / n))
        print(f'{line} [{time.time()-t0:.0f}s]', flush=True)
    torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': best}, f'{outdir}/last.pt')
    if not va:
        torch.save({'model': model.state_dict(), 'args': vars(args), 'rmse': -1}, f'{outdir}/best.pt')
    print(f'DONE fold {args.fold} best val RMSE {best:.2f}  time {(time.time()-t0)/60:.1f}min', flush=True)
    with open(f'{outdir}/result.json', 'w') as f:
        json.dump({'best_rmse': best, 'args': vars(args)}, f)


if __name__ == '__main__':
    main()
