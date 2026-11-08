import argparse
import os
import time
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
from common import WORK, UNet, UNetV2, bce_dice_loss, predict_full, dice, load_split, resize_to_512
from train import get_folds, make_padded, make_batch, ModelEMA

cv2.setNumThreads(1)


@torch.no_grad()
def evaluate(model, val_ids, imgs6, labs, fovs, thrs=(0.4, 0.5, 0.6)):
    model.eval()
    res = {t: [] for t in thrs}
    for i in val_ids:
        img = imgs6[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        prob = predict_full(model, img, tta=False) * fov
        for t in thrs:
            res[t].append(dice(resize_to_512((prob > t).astype(np.uint8)), resize_to_512(labs[i])))
    return {t: float(np.mean(v)) for t, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, default=192)
    ap.add_argument('--crop', type=int, default=288)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--blobs', type=float, default=0.2)
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', type=str, default='')
    ap.add_argument('--arch', type=str, default='v1', choices=['v1', 'v2'])
    ap.add_argument('--oof', nargs='+', default=[os.path.join(WORK, 'oof_best_03.npz'),
                                                 os.path.join(WORK, 'oof_best_4.npz')])
    ap.add_argument('--frangi', type=str, default='')
    ap.add_argument('--out', type=str, default=os.path.join(WORK, 'runs_refiner'))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(77 + args.fold + args.seed * 1000)
    np.random.seed(77 + args.fold + args.seed * 1000)
    rng = np.random.RandomState(2024 + args.fold + args.seed * 100)

    folds = get_folds()
    val_ids = folds[args.fold]
    train_ids = [i for i in range(1, 21) if i not in val_ids]
    imgs, labs, fovs = load_split(range(1, 21), 'train')
    oof = {}
    for p in args.oof:
        d = np.load(p)
        for k in d.files:
            oof[int(k)] = d[k].astype(np.float32)
    imgs6 = {}
    for i in range(1, 21):
        im = imgs[i].astype(np.float32)
        parts = [im[:4], oof[i][None]]
        if args.frangi:
            v = np.load(os.path.join(args.frangi, f'{i}.npy')).astype(np.float32)
            sel = fovs[i] > 0
            hi = np.percentile(v[sel], 99.5) + 1e-6
            parts.append(np.clip(v / hi, 0, 1)[None])
        parts.append(im[4:5])
        imgs6[i] = np.concatenate(parts, axis=0).astype(np.float16)
    fov_ch = imgs6[1].shape[0] - 1
    print('fold', args.fold, 'train', train_ids, 'val', val_ids, 'in_ch', imgs6[1].shape[0], flush=True)

    kw = dict(crop=args.crop, out=args.patch, n_img=4, fov_ch=fov_ch, blob_p=args.blobs)
    data = make_padded(train_ids, imgs6, labs, fovs)
    outdir = os.path.join(args.out, f'f{args.fold}')
    os.makedirs(outdir, exist_ok=True)

    in_ch = imgs6[1].shape[0]
    net_cls = UNetV2 if args.arch == 'v2' else UNet
    model = net_cls(in_ch=in_ch, base=args.base)
    if args.init:
        ck0 = torch.load(args.init, map_location='cpu', weights_only=False)
        model.load_state_dict(ck0['model'])
        print('init from', args.init, flush=True)
    print('params %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 100

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p))) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = ModelEMA(model, args.ema) if args.ema > 0 else None
    ema_model = net_cls(in_ch=in_ch, base=args.base) if ema is not None else None
    logf = open(os.path.join(outdir, 'log.csv'), 'w', newline='')
    writer = csv.writer(logf)
    best = -1
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(data, train_ids, rng, args.batch, **kw)
        logits = model(x)
        loss = bce_dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update(model)
        if step % 50 == 0:
            print('step %d loss %.4f %.0fs' % (step, loss.item(), time.time() - t0), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, val_ids, imgs6, labs, fovs)
            print('EVAL', step, {k: round(v, 4) for k, v in ev.items()}, flush=True)
            writer.writerow([step] + [ev[t] for t in (0.4, 0.5, 0.6)])
            logf.flush()
            score = max(ev.values())
            cand, cand_ema = model, False
            if ema is not None:
                ema_model.load_state_dict(ema.shadow)
                ev2 = evaluate(ema_model, val_ids, imgs6, labs, fovs)
                print('EVAL_EMA', step, {k: round(v, 4) for k, v in ev2.items()}, flush=True)
                s2 = max(ev2.values())
                if s2 > score:
                    score, cand, cand_ema = s2, ema_model, True
            if score > best:
                best = score
                torch.save({'model': cand.state_dict(), 'fold': args.fold, 'step': step, 'val_dice': best,
                            'arch': args.arch, 'base': args.base, 'ema': cand_ema}, os.path.join(outdir, 'best.pt'))
            model.train()
    torch.save({'model': model.state_dict(), 'fold': args.fold, 'step': args.steps, 'val_dice': best,
                'arch': args.arch, 'base': args.base}, os.path.join(outdir, 'last.pt'))
    logf.close()
    print('done fold %d best %.4f total %.0fs' % (args.fold, best, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
