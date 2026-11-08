import argparse
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from common import WORK, UNet, bce_dice_loss, predict_full, resize_to_512, dice, load_split
from train import get_folds, make_padded, sample_pair, ModelEMA

cv2.setNumThreads(1)


def sample_test_patch(tim, tprob, fovp, rng, crop=288, out=192, pad=192):
    H, W = tim.shape[1:]
    H0, W0 = H - 2 * pad, W - 2 * pad
    ys, xs = np.nonzero(fovp[pad:pad + H0, pad:pad + W0] > 0.5)
    if len(ys) > 0:
        k = rng.randint(len(ys))
        cy, cx = int(ys[k]), int(xs[k])
    else:
        cy, cx = H0 // 2, W0 // 2
    y0 = cy + pad - crop // 2
    x0 = cx + pad - crop // 2
    ip = tim[:, y0:y0 + crop, x0:x0 + crop].copy()
    tp = tprob[y0:y0 + crop, x0:x0 + crop].copy()
    fp = fovp[y0:y0 + crop, x0:x0 + crop].copy()

    ang = rng.uniform(-180, 180)
    sc = rng.uniform(0.8, 1.3)
    M = cv2.getRotationMatrix2D((crop / 2, crop / 2), ang, sc)
    M[0, 2] += rng.uniform(-crop * 0.09, crop * 0.09)
    M[1, 2] += rng.uniform(-crop * 0.09, crop * 0.09)
    ip = cv2.warpAffine(ip.transpose(1, 2, 0), M, (out, out), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101).transpose(2, 0, 1)
    tp = cv2.warpAffine(tp, M, (out, out), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fp = cv2.warpAffine(fp, M, (out, out), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        ip = ip[:, :, ::-1]; tp = tp[:, ::-1]; fp = fp[:, ::-1]
    if rng.random() < 0.5:
        ip = ip[:, ::-1, :]; tp = tp[::-1, :]; fp = fp[::-1, :]
    if rng.random() < 0.25:
        yy, xx = np.meshgrid(np.arange(out, dtype=np.float32), np.arange(out, dtype=np.float32))
        dx = cv2.GaussianBlur(rng.normal(0, 1, (out, out)).astype(np.float32), (0, 0), 8) * 6
        dy = cv2.GaussianBlur(rng.normal(0, 1, (out, out)).astype(np.float32), (0, 0), 8) * 6
        mx = (xx + dx).astype(np.float32); my = (yy + dy).astype(np.float32)
        ip = cv2.remap(ip.transpose(1, 2, 0), mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101).transpose(2, 0, 1)
        tp = cv2.remap(tp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fp = cv2.remap(fp, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    g = rng.uniform(0.85, 1.15, (4, 1, 1)).astype(np.float32)
    b = rng.uniform(-0.15, 0.15, (4, 1, 1)).astype(np.float32)
    ip = ip.copy()
    ip[:4] = ip[:4] * g + b
    if rng.random() < 0.3:
        ip[:4] += rng.normal(0, 0.05, (4, out, out)).astype(np.float32)
    ip[4] = fp
    return ip, tp[None]


def eval_dice512(model, val_ids, imgs, labs, fovs):
    model.eval()
    vals = []
    for i in val_ids:
        img = imgs[i].astype(np.float32)
        fov = fovs[i].astype(np.float32)
        p = predict_full(model, img, tta=False) * fov
        vals.append(dice(resize_to_512((p > 0.5).astype(np.uint8)), resize_to_512(labs[i])))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, default=192)
    ap.add_argument('--crop', type=int, default=288)
    ap.add_argument('--pad', type=int, default=192)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--teacher-dir', default=os.path.join(WORK, 'submission_refined_ens', 'prob'))
    ap.add_argument('--init-run', default=os.path.join(WORK, 'runs_v1'))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--allreal', action='store_true')
    ap.add_argument('--ema', type=float, default=0.99)
    ap.add_argument('--out', default=os.path.join(WORK, 'runs_ft'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(555 + args.fold)
    np.random.seed(555 + args.fold)
    rng = np.random.RandomState(555 + args.fold)

    folds = get_folds()
    if args.all:
        val_ids = []
        train_ids = list(range(1, 21))
    else:
        val_ids = folds[args.fold]
        train_ids = [i for i in range(1, 21) if i not in val_ids]
    imgs, labs, fovs = load_split(range(1, 21), 'train')
    timgs, _, tfovs = load_split(range(1, 21), 'test')

    data = make_padded(train_ids, imgs, labs, fovs, pad=args.pad)
    tim_p, tp_p, tfov_p = {}, {}, {}
    for i in range(1, 21):
        tim = np.pad(timgs[i].astype(np.float32), ((0, 0), (args.pad, args.pad), (args.pad, args.pad)),
                     mode='reflect')
        tprob = np.load(os.path.join(args.teacher_dir, f'{i}.npy')).astype(np.float32)
        tprob = np.pad(tprob, ((args.pad, args.pad), (args.pad, args.pad)), mode='edge')
        tim_p[i] = tim
        tp_p[i] = tprob
        tfov_p[i] = np.pad(tfovs[i].astype(np.float32), ((args.pad, args.pad), (args.pad, args.pad)))

    outdir = os.path.join(args.out, f'f{args.fold}')
    os.makedirs(outdir, exist_ok=True)
    model = UNet(in_ch=imgs[1].shape[0], base=args.base)
    init_step = None
    if args.init_run:
        ck = torch.load(os.path.join(args.init_run, f'f{args.fold}_s1', 'best.pt'), map_location='cpu',
                        weights_only=False)
        model.load_state_dict(ck['model'])
        init_step = ck.get('step')
    model.train()
    print('fold', args.fold, 'val', val_ids, 'init step', init_step, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    ema = ModelEMA(model, args.ema) if args.ema > 0 else None
    ema_model = UNet(in_ch=imgs[1].shape[0], base=args.base) if ema is not None else None
    best = -1
    t0 = time.time()
    for step in range(1, args.steps + 1):
        xs, ys = [], []
        n_test = 0 if args.allreal else args.batch - args.batch // 2
        for _ in range(args.batch - n_test):
            ip, lp, vp = sample_pair(data, train_ids, rng, crop=args.crop, out=args.patch, pad=args.pad)
            xs.append(ip); ys.append(lp)
        for _ in range(n_test):
            i = rng.randint(1, 21)
            ip, tp = sample_test_patch(tim_p[i], tp_p[i], tfov_p[i], rng, crop=args.crop, out=args.patch,
                                       pad=args.pad)
            xs.append(ip); ys.append(tp[0])
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))[:, None]
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
        if (step % args.eval_every == 0 or step == args.steps) and len(val_ids) > 0:
            d = eval_dice512(model, val_ids, imgs, labs, fovs)
            print('EVAL512', step, round(d, 4), flush=True)
            cand, is_ema = model, False
            if ema is not None:
                ema_model.load_state_dict(ema.shadow)
                d2 = eval_dice512(ema_model, val_ids, imgs, labs, fovs)
                print('EVAL512_EMA', step, round(d2, 4), flush=True)
                if d2 > d:
                    d, cand, is_ema = d2, ema_model, True
            if d > best:
                best = d
                torch.save({'model': cand.state_dict(), 'fold': args.fold, 'step': step, 'val_dice': best,
                            'arch': 'v1', 'base': args.base, 'ema': is_ema}, os.path.join(outdir, 'best.pt'))
            model.train()
    torch.save({'model': model.state_dict(), 'fold': args.fold, 'step': args.steps, 'val_dice': best,
                'arch': 'v1', 'base': args.base}, os.path.join(outdir, 'last.pt'))
    print('done fold %d best %.4f total %.0fs' % (args.fold, best, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
