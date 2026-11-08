import argparse
import os
import time
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.morphology import skeletonize
from common import CACHE, WORK, UNet, UNetV2, bce_dice_loss, tversky_loss, predict_full, dice, load_split
from losses import cl_dice_loss

cv2.setNumThreads(1)


def get_folds(seed=123):
    ids = np.arange(1, 21)
    perm = np.random.RandomState(seed).permutation(ids)
    return [sorted(perm[i::5].tolist()) for i in range(5)]


def make_padded(ids, imgs, labs, fovs, pad=192, valids=None):
    out = {}
    for i in ids:
        im = imgs[i].astype(np.float32)
        lb = labs[i].astype(np.float32)
        fo = fovs[i].astype(np.float32)
        va = np.ones_like(lb) if valids is None else valids[i].astype(np.float32)
        imp = np.pad(im, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
        lbp = np.pad(lb, ((pad, pad), (pad, pad)), mode='constant')
        fop = np.pad(fo, ((pad, pad), (pad, pad)), mode='constant')
        vap = np.pad(va, ((pad, pad), (pad, pad)), mode='constant')
        ys, xs = np.nonzero(lb > 0.5)
        out[i] = (imp, lbp, fop, ys, xs, vap)
    return out


_GRID = {}


def _grid(n):
    if n not in _GRID:
        yy, xx = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
        _GRID[n] = (xx, yy)
    return _GRID[n]


def sample_pair(data, ids, rng, crop=352, out=256, pad=192, blob_p=0.0, full=False, n_img=4, fov_ch=4,
                return_aux=False):
    i = ids[rng.randint(len(ids))]
    imp, lbp, fop, ys, xs, vap = data[i]
    H, W = lbp.shape
    H0, W0 = H - 2 * pad, W - 2 * pad
    if full:
        cy, cx = H0 // 2, W0 // 2
    elif rng.random() < 0.5 and len(ys) > 0:
        k = rng.randint(len(ys))
        cy, cx = int(ys[k]), int(xs[k])
    else:
        k = rng.randint(H0 * W0)
        cy, cx = divmod(k, W0)
    y0 = cy + pad - crop // 2
    x0 = cx + pad - crop // 2
    ip = imp[:, y0:y0 + crop, x0:x0 + crop]
    lp = lbp[y0:y0 + crop, x0:x0 + crop]
    fp = fop[y0:y0 + crop, x0:x0 + crop]
    vp = vap[y0:y0 + crop, x0:x0 + crop]

    # random affine: rotation + scale + translation
    ang = rng.uniform(-180, 180)
    sc = rng.uniform(0.8, 1.3)
    M = cv2.getRotationMatrix2D((crop / 2, crop / 2), ang, sc)
    M[0, 2] += rng.uniform(-crop * 0.09, crop * 0.09)
    M[1, 2] += rng.uniform(-crop * 0.09, crop * 0.09)
    ip = cv2.warpAffine(ip.transpose(1, 2, 0), M, (out, out), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101).transpose(2, 0, 1)
    lp = cv2.warpAffine(lp, M, (out, out), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fp = cv2.warpAffine(fp, M, (out, out), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vp = cv2.warpAffine(vp, M, (out, out), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if rng.random() < 0.5:
        ip = ip[:, :, ::-1]
        lp = lp[:, ::-1]
        fp = fp[:, ::-1]
        vp = vp[:, ::-1]
    if rng.random() < 0.5:
        ip = ip[:, ::-1, :]
        lp = lp[::-1, :]
        fp = fp[::-1, :]
        vp = vp[::-1, :]

    # elastic
    if rng.random() < 0.25:
        xx, yy = _grid(out)
        dx = cv2.GaussianBlur(rng.normal(0, 1, (out, out)).astype(np.float32), (0, 0), 8) * 6
        dy = cv2.GaussianBlur(rng.normal(0, 1, (out, out)).astype(np.float32), (0, 0), 8) * 6
        mx = (xx + dx).astype(np.float32)
        my = (yy + dy).astype(np.float32)
        ip = cv2.remap(ip.transpose(1, 2, 0), mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101).transpose(2, 0, 1)
        lp = cv2.remap(lp, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fp = cv2.remap(fp, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        vp = cv2.remap(vp, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # photometric on the image channels only
    g = rng.uniform(0.85, 1.15, (n_img, 1, 1)).astype(np.float32)
    b = rng.uniform(-0.15, 0.15, (n_img, 1, 1)).astype(np.float32)
    ip = ip.copy()
    ip[:n_img] = ip[:n_img] * g + b
    if rng.random() < 0.3:
        ip[:n_img] += rng.normal(0, 0.05, (n_img, out, out)).astype(np.float32)
    if rng.random() < 0.15:
        s = rng.uniform(0.5, 1.2)
        for c in range(n_img):
            ip[c] = cv2.GaussianBlur(ip[c], (3, 3), s)

    # pathology-like smooth bright/dark blobs (exudates / haemorrhages)
    if blob_p > 0 and rng.random() < blob_p:
        blob = np.zeros((out, out), np.float32)
        for _ in range(rng.randint(1, 4)):
            cx, cy = rng.randint(0, out), rng.randint(0, out)
            rx, ry = rng.randint(6, 55), rng.randint(6, 55)
            cv2.ellipse(blob, (cx, cy), (rx, ry), rng.uniform(0, 180), 0, 360, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), rng.uniform(3.0, 9.0))
        amp = rng.uniform(-1.4, 1.6)
        mix = rng.uniform(0.6, 1.2, (n_img, 1, 1)).astype(np.float32)
        ip[:n_img] += blob[None] * amp * mix

    # cutout / coarse dropout
    if rng.random() < 0.25:
        for _ in range(rng.randint(1, 3)):
            bh = rng.randint(10, 48)
            bw = rng.randint(10, 48)
            by = rng.randint(0, out - bh)
            bx = rng.randint(0, out - bw)
            ip[:n_img, by:by + bh, bx:bx + bw] = rng.normal(0, 0.5, (n_img, bh, bw)).astype(np.float32)
            lp[by:by + bh, bx:bx + bw] = 0
            vp[by:by + bh, bx:bx + bw] = 0

    ip[fov_ch] = fp
    if return_aux:
        binm = (lp > 0.5).astype(np.uint8)
        skel = skeletonize(binm > 0).astype(np.float32)
        bnd = (cv2.morphologyEx(binm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0).astype(np.float32)
        return ip, lp, vp, np.stack([skel, bnd], axis=0)
    return ip, lp, vp


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def make_batch(data, ids, rng, batch, with_weight=False, **kw):
    xs, ys, ws = [], [], []
    for _ in range(batch):
        ip, lp, vp = sample_pair(data, ids, rng, **kw)
        xs.append(ip)
        ys.append(lp)
        ws.append(vp)
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))[:, None]
    if not with_weight:
        return x, y
    w = torch.from_numpy(np.stack(ws))[:, None]
    return x, y, w


@torch.no_grad()
def evaluate(model, val_ids, imgs, labs, fovs, thrs=(0.4, 0.5, 0.6), tta=False):
    model.eval()
    res = {t: [] for t in thrs}
    for i in val_ids:
        img = imgs[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        prob = predict_full(model, img, tta=tta) * fov
        for t in thrs:
            res[t].append(dice(prob > t, labs[i]))
    return {t: float(np.mean(v)) for t, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--crop', type=int, default=352)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--cldice', type=float, default=0.0)
    ap.add_argument('--select', type=str, default='05', choices=['05', 'max'])
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--pseudo', type=str, default='')
    ap.add_argument('--real-repeat', type=int, default=2)
    ap.add_argument('--blobs', type=float, default=0.0)
    ap.add_argument('--arch', type=str, default='v1', choices=['v1', 'v2'])
    ap.add_argument('--tversky', type=float, default=0.0)
    ap.add_argument('--pad', type=int, default=192)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--cache', type=str, default='')
    ap.add_argument('--split-seed', type=int, default=123)
    ap.add_argument('--out', type=str, default=os.path.join(WORK, 'runs'))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed + args.fold * 100)
    np.random.seed(args.seed + args.fold * 100)
    rng = np.random.RandomState(1000 + args.seed * 10 + args.fold)

    folds = get_folds(args.split_seed)
    val_ids = folds[args.fold]
    train_ids = [i for i in range(1, 21) if i not in val_ids]
    print('fold', args.fold, 'train', train_ids, 'val', val_ids, flush=True)

    cache_dir = args.cache or None
    imgs, labs, fovs = load_split(range(1, 21), 'train', cache=cache_dir)
    sample_ids = list(train_ids)
    valids = None
    if args.pseudo:
        pdir = os.path.join(args.pseudo, f'f{args.fold}')
        timgs, _, tfovs = load_split(range(1, 21), 'test', cache=cache_dir)
        valids = {i: np.ones_like(labs[i], np.uint8) for i in train_ids}
        tids = []
        for i in range(1, 21):
            k = 100 + i
            d = np.load(os.path.join(pdir, f'{i}.npz'))
            imgs[k] = timgs[i]
            fovs[k] = tfovs[i]
            labs[k] = d['label'].astype(np.uint8)
            valids[k] = d['valid'].astype(np.uint8)
            tids.append(k)
        sample_ids = list(train_ids) * args.real_repeat + tids
    data = make_padded(sample_ids, imgs, labs, fovs, pad=args.pad, valids=valids)
    outdir = os.path.join(args.out, f'f{args.fold}_s{args.seed}')
    os.makedirs(outdir, exist_ok=True)

    net_cls = UNetV2 if args.arch == 'v2' else UNet
    in_ch = imgs[1].shape[0]
    model = net_cls(in_ch=in_ch, base=args.base)
    nparam = sum(p.numel() for p in model.parameters())
    print('params %.2fM in_ch %d' % (nparam / 1e6, in_ch), flush=True)
    ema = ModelEMA(model, args.ema) if args.ema > 0 else None
    ema_model = net_cls(in_ch=in_ch, base=args.base) if ema is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 100

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p))) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    logf = open(os.path.join(outdir, 'log.csv'), 'w', newline='')
    writer = csv.writer(logf)
    writer.writerow(['step', 'loss', 'val04', 'val05', 'val06', 'sec'])
    best = -1
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        if valids is None:
            x, y = make_batch(data, sample_ids, rng, args.batch, crop=args.crop, out=args.patch,
                              blob_p=args.blobs, pad=args.pad, full=args.full)
            logits = model(x)
            loss = bce_dice_loss(logits, y)
        else:
            x, y, w = make_batch(data, sample_ids, rng, args.batch, with_weight=True,
                                 crop=args.crop, out=args.patch, blob_p=args.blobs, pad=args.pad,
                                 full=args.full)
            logits = model(x)
            loss = bce_dice_loss(logits, y, w)
        if args.cldice > 0:
            loss = loss + args.cldice * cl_dice_loss(logits, y)
        if args.tversky > 0:
            loss = loss + args.tversky * tversky_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update(model)
        if step % 50 == 0:
            print('step %d loss %.4f lr %.2e %.0fs' % (step, loss.item(), sched.get_last_lr()[0], time.time() - t0), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, val_ids, imgs, labs, fovs)
            line = [step, round(loss.item(), 5), round(ev[0.4], 5), round(ev[0.5], 5), round(ev[0.6], 5), round(time.time() - t0, 1)]
            writer.writerow(line)
            logf.flush()
            print('EVAL', line, flush=True)
            score = ev[0.5] if args.select == '05' else max(ev.values())
            cand = model
            cand_ema = False
            if ema is not None:
                ema_model.load_state_dict(ema.shadow)
                ev_ema = evaluate(ema_model, val_ids, imgs, labs, fovs)
                line_ema = ['ema', line[1], round(ev_ema[0.4], 5), round(ev_ema[0.5], 5),
                            round(ev_ema[0.6], 5), line[-1]]
                print('EVAL_EMA', line_ema, flush=True)
                s2 = ev_ema[0.5] if args.select == '05' else max(ev_ema.values())
                if s2 > score:
                    score = s2
                    cand = ema_model
                    cand_ema = True
            if score > best:
                best = score
                torch.save({'model': cand.state_dict(), 'fold': args.fold, 'seed': args.seed, 'step': step,
                            'val_dice': best, 'cldice': args.cldice, 'select': args.select, 'ema': cand_ema,
                            'arch': args.arch, 'base': args.base},
                           os.path.join(outdir, 'best.pt'))
            model.train()
    torch.save({'model': model.state_dict(), 'fold': args.fold, 'seed': args.seed, 'step': args.steps,
                'val_dice': best, 'arch': args.arch, 'base': args.base}, os.path.join(outdir, 'last.pt'))
    if ema is not None:
        ema_model.load_state_dict(ema.shadow)
        torch.save({'model': ema_model.state_dict(), 'fold': args.fold, 'seed': args.seed, 'step': args.steps,
                    'val_dice': evaluate(ema_model, val_ids, imgs, labs, fovs)[0.5], 'arch': args.arch,
                    'base': args.base, 'ema': True},
                   os.path.join(outdir, 'last_ema.pt'))
    logf.close()
    print('done fold %d best %.4f total %.0fs' % (args.fold, best, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
