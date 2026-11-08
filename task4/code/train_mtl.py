import argparse
import os
import time
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import WORK, MTLUNet, bce_dice_loss, predict_full, dice, load_split
from losses import cl_dice_loss
from train import get_folds, make_padded, sample_pair, evaluate, ModelEMA

cv2.setNumThreads(1)


def make_batch_mtl(data, ids, rng, batch, **kw):
    xs, ys, auxs = [], [], []
    for _ in range(batch):
        ip, lp, vp, aux = sample_pair(data, ids, rng, return_aux=True, **kw)
        xs.append(ip)
        ys.append(lp)
        auxs.append(aux)
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))[:, None]
    a = torch.from_numpy(np.stack(auxs))
    return x, y, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, default=192)
    ap.add_argument('--crop', type=int, default=288)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--blobs', type=float, default=0.2)
    ap.add_argument('--ema', type=float, default=0.995)
    ap.add_argument('--cldice', type=float, default=0.2)
    ap.add_argument('--aux-w', type=float, default=0.25)
    ap.add_argument('--cache', type=str, default='')
    ap.add_argument('--out', type=str, default=os.path.join(WORK, 'runs_mtl'))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(303 + args.fold)
    np.random.seed(303 + args.fold)
    rng = np.random.RandomState(303 + args.fold)

    folds = get_folds()
    val_ids = folds[args.fold]
    train_ids = [i for i in range(1, 21) if i not in val_ids]
    imgs, labs, fovs = load_split(range(1, 21), 'train', cache=args.cache or None)
    data = make_padded(train_ids, imgs, labs, fovs)
    outdir = os.path.join(args.out, f'f{args.fold}')
    os.makedirs(outdir, exist_ok=True)
    print('fold', args.fold, 'train', train_ids, 'val', val_ids, flush=True)

    model = MTLUNet(in_ch=imgs[1].shape[0], base=args.base)
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
    ema_model = MTLUNet(in_ch=imgs[1].shape[0], base=args.base) if ema is not None else None
    logf = open(os.path.join(outdir, 'log.csv'), 'w', newline='')
    writer = csv.writer(logf)
    best = -1
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y, a = make_batch_mtl(data, train_ids, rng, args.batch, crop=args.crop, out=args.patch,
                                 blob_p=args.blobs)
        seg, aux = model(x)
        loss = bce_dice_loss(seg, y)
        if args.cldice > 0:
            loss = loss + args.cldice * cl_dice_loss(seg, y)
        aux_loss = F.binary_cross_entropy_with_logits(aux[:, 0:1], a[:, 0:1],
                                                      pos_weight=torch.tensor(5.0))
        aux_loss = aux_loss + F.binary_cross_entropy_with_logits(aux[:, 1:2], a[:, 1:2],
                                                                 pos_weight=torch.tensor(2.0))
        loss = loss + args.aux_w * aux_loss
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
            ev = evaluate(model, val_ids, imgs, labs, fovs)
            print('EVAL', step, {k: round(v, 4) for k, v in ev.items()}, flush=True)
            writer.writerow([step] + [ev[t] for t in (0.4, 0.5, 0.6)])
            logf.flush()
            score = max(ev.values())
            cand, cand_ema = model, False
            if ema is not None:
                ema_model.load_state_dict(ema.shadow)
                ev2 = evaluate(ema_model, val_ids, imgs, labs, fovs)
                print('EVAL_EMA', step, {k: round(v, 4) for k, v in ev2.items()}, flush=True)
                s2 = max(ev2.values())
                if s2 > score:
                    score, cand, cand_ema = s2, ema_model, True
            if score > best:
                best = score
                torch.save({'model': cand.state_dict(), 'fold': args.fold, 'step': step, 'val_dice': best,
                            'arch': 'mtl', 'base': args.base, 'ema': cand_ema}, os.path.join(outdir, 'best.pt'))
            model.train()
    torch.save({'model': model.state_dict(), 'fold': args.fold, 'step': args.steps, 'val_dice': best,
                'arch': 'mtl', 'base': args.base}, os.path.join(outdir, 'last.pt'))
    logf.close()
    print('done fold %d best %.4f total %.0fs' % (args.fold, best, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
