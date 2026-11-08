#!/usr/bin/env python3
"""FER 6-class emotion classification -- k-fold CNN ensemble.

Constraints honoured: only the competition-provided data is used.  ImageNet
initialisation is used only as an initialisation (explicitly allowed by the
task description) -- no external emotion labels are ever consulted.

Outputs (written to --out):
  fold{k}_oof_{raw,ema}.npy    (n_train, 6) out-of-fold probabilities
  fold{k}_test_{raw,ema}.npy   (n_test, 6)  test probabilities (TTA)
  fold_ids.npy                 (n_train,)   fold index of every train sample
  summary.json                 run metadata / scores
  submission_<run>.csv         ID,Emotion using the mean over finished folds
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T

# Official label ids from the competition rules.
CLASS_ORDER = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
N_CLASSES = len(CLASS_ORDER)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def heartbeat(out_dir: str, **kw):
    """Tiny progress file: lets the caller detect a stalled/silently-CPU run early."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "heartbeat.json"), "w") as f:
            json.dump({"time": time.strftime("%H:%M:%S"), **kw}, f)
    except Exception:
        pass


# --------------------------------------------------------------------------- data
def find_data_root() -> str:
    env_root = os.environ.get("FER_DATA_ROOT", "").strip()
    if env_root and os.path.isdir(os.path.join(env_root, "train")) and os.path.isdir(os.path.join(env_root, "test")):
        return env_root
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        "/kaggle/input/competitions/2026task2/fer_data/fer_data",
        "/kaggle/input/competitions/2026task2/fer_data",
        "/kaggle/input/competitions/2026task2",
        "/kaggle/input/2026task2/fer_data/fer_data",
        "/kaggle/input/fer_data/fer_data",
        "/kaggle/input/2026task2/fer_data",
        "/kaggle/input/2026task2",
        os.path.join(here, "..", "fer_data", "fer_data"),
        os.path.join(here, "fer_data", "fer_data"),
    ]

    def ok(p):
        return (
            os.path.isdir(os.path.join(p, "train"))
            and os.path.isdir(os.path.join(p, "test"))
            and len(os.listdir(os.path.join(p, "test"))) > 10
        )

    for c in cands:
        if ok(c):
            return c
    # bounded crawl of the mounted inputs
    for base in ("/kaggle/input", "."):
        if not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            if root.count(os.sep) - base.count(os.sep) > 5:
                dirs[:] = []
                continue
            if ok(root):
                return root
    # zipped input fallback
    import zipfile

    for base in ("/kaggle/input", "."):
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".zip"):
                    dst = "/kaggle/working/_data" if os.path.isdir("/kaggle/working") else "./_data"
                    os.makedirs(dst, exist_ok=True)
                    log("extracting", os.path.join(root, f), "->", dst)
                    with zipfile.ZipFile(os.path.join(root, f)) as z:
                        z.extractall(dst)
                    for r2, d2, _ in os.walk(dst):
                        if ok(r2):
                            return r2
    raise FileNotFoundError("competition data not found")


def resolve_file(path: str) -> str:
    """Locate an auxiliary input (kernel sources, working dir or mounted inputs)."""
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    cands = ["/kaggle/src", "/kaggle/working", os.path.dirname(os.path.abspath(__file__)), "."]
    for c in cands:
        p = os.path.join(c, base)
        if os.path.exists(p):
            return p
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) > 6:
            dirs[:] = []
            continue
        if base in files:
            return os.path.join(root, base)
    raise FileNotFoundError(f"auxiliary file not found: {path}")


def load_split(root: str):
    """Returns (X_train uint8 [n,48,48], y [n], X_test uint8 [m,48,48], names, n_per_class)."""
    from PIL import Image

    tr = os.path.join(root, "train")
    per_class = []
    Xs, ys = [], []
    for ci, cname in enumerate(CLASS_ORDER):
        cdir = os.path.join(tr, cname)
        files = sorted(os.listdir(cdir))
        per_class.append(len(files))
        for f in files:
            im = Image.open(os.path.join(cdir, f)).convert("L")
            Xs.append(np.asarray(im, dtype=np.uint8))
            ys.append(ci)
    te = os.environ.get("FER_TEST_ROOT", "").strip() or os.path.join(root, "test")
    if os.environ.get("FER_TEST_ROOT", "").strip():
        log(f"using FER_TEST_ROOT override: {te}")
    names = sorted(os.listdir(te))
    Xt = [np.asarray(Image.open(os.path.join(te, f)).convert("L"), dtype=np.uint8) for f in names]
    Xs = np.stack(Xs)[:, None]
    Xt = np.stack(Xt)[:, None]
    return Xs, np.array(ys, np.int64), Xt, names, per_class


# --------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.b2(self.c2(x))
        return F.relu(x + idt, inplace=True)


class SmallResNet(nn.Module):
    """ResNet tuned for 48x48 grayscale faces (trains from scratch quickly)."""

    def __init__(self, width=64, blocks=(2, 2, 2, 2), p_drop=0.3, n_classes=N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)
        )
        layers = []
        cin = width
        for i, nb in enumerate(blocks):
            cout = width * (2 ** i)
            for b in range(nb):
                stride = 2 if (b == 0 and i > 0) else 1
                layers.append(BasicBlock(cin, cout, stride))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(cin, n_classes))

    def forward(self, x):
        x = self.body(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def build_model(arch: str, size: int, pretrained: bool, width: int = 64):
    """Returns (net, used_pretrained_weights, input_channels_expected)."""
    if arch.startswith("scratch"):
        return SmallResNet(width=width), False, 1
    if arch.startswith("tv_"):
        import torchvision.models as tvm

        name = arch[3:]
        net, used_pret = None, False
        if pretrained:
            try:
                net = getattr(tvm, name)(weights="DEFAULT")
                used_pret = True
                log(f"loaded torchvision {name} with ImageNet weights")
            except Exception as e:  # pragma: no cover - offline fallback
                log(f"pretrained {name} unavailable ({e}); random init")
        if net is None:
            net = getattr(tvm, name)(weights=None)
        if hasattr(net, "fc"):
            net.fc = nn.Linear(net.fc.in_features, N_CLASSES)
        elif hasattr(net, "classifier"):
            cl = net.classifier
            if isinstance(cl, nn.Sequential):
                cl[-1] = nn.Linear(cl[-1].in_features, N_CLASSES)
            else:
                net.classifier = nn.Linear(cl.in_features, N_CLASSES)
        return net, used_pret, 3
    if arch.startswith("timm_"):
        import timm

        name = arch[5:]
        net = timm.create_model(name, pretrained=pretrained, num_classes=N_CLASSES, in_chans=3)
        return net, bool(pretrained), 3
    raise ValueError(f"unknown arch {arch}")


# --------------------------------------------------------------------------- augmentation
def make_aug(size: int, strong: bool):
    import torchvision.transforms.v2 as T

    ops = []
    if strong:
        ops += [
            T.RandomApply(
                [T.RandomRotation(12, interpolation=T.InterpolationMode.BILINEAR, fill=0.0)], p=0.7
            ),
            T.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1),
                           interpolation=T.InterpolationMode.BILINEAR, fill=0.0),
        ]
    ops += [
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.85, 1.15),
                            interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.RandomHorizontalFlip(0.5),
    ]
    geo = T.Compose(ops)
    # NOTE: RandomErasing(->cutout) is applied after normalisation in the loop.

    def aug(x_u8):  # (B,1,48,48) uint8 on device -> (B,1,size,size) float in [0,1]
        x = x_u8.float().div_(255.0)
        return geo(x)

    return aug


_ERASE = None


def random_erasing(x):
    global _ERASE
    if _ERASE is None:
        _ERASE = T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0)
    x = _ERASE(x)
    return T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=1.0)(x)


class FastAug:
    """Vectorised GPU augmentation: one affine warp + photometric jitter + cutout.

    All operations are single tensor kernels, which keeps the data pipeline off the
    critical path (the per-sample transforms of torchvision cost several times more).
    """

    def __init__(self, size, strong=True, rot=12.0, scale=(0.82, 1.15), trans=0.08,
                 p_cutout=0.5):
        self.size = size
        self.rot = rot if strong else 0.0
        self.scale = scale if strong else (1.0, 1.0)
        self.trans = trans if strong else 0.0
        self.p_cutout = p_cutout if strong else 0.0
        self._grid_cache = None

    def __call__(self, x_u8):
        x = x_u8.float().div_(255.0)
        B = x.shape[0]
        dev, S = x.device, self.size
        rr = torch.rand(B, device=dev)
        ang = (rr * 2 - 1) * (self.rot * math.pi / 180.0)
        sc = torch.empty(B, device=dev).uniform_(*self.scale)
        cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
        tx = (torch.rand(B, device=dev) * 2 - 1) * self.trans * 2
        ty = (torch.rand(B, device=dev) * 2 - 1) * self.trans * 2
        theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
        grid = F.affine_grid(theta, (B, 1, S, S), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        flip = torch.rand(B, device=dev) < 0.5
        if bool(flip.any()):
            x[flip] = torch.flip(x[flip], dims=[3])
        a = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.12
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.10
        x = (x * a + b).clamp_(0.0, 1.0)
        if self.p_cutout > 0:
            m = torch.rand(B, device=dev) < self.p_cutout
            if bool(m.any()):
                if self._grid_cache is None or self._grid_cache[0] != (S, dev):
                    yy, xx = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev),
                                            indexing="ij")
                    self._grid_cache = ((S, dev), yy, xx)
                _, yy, xx = self._grid_cache
                h = (torch.rand(B, device=dev) * 0.20 + 0.05) * S
                w = (torch.rand(B, device=dev) * 0.20 + 0.05) * S
                y0 = torch.rand(B, device=dev) * (S - h)
                x0 = torch.rand(B, device=dev) * (S - w)
                box = ((yy[None] >= y0[:, None, None]) & (yy[None] < (y0 + h)[:, None, None]) &
                       (xx[None] >= x0[:, None, None]) & (xx[None] < (x0 + w)[:, None, None]))
                box &= m[:, None, None]
                x = x.masked_fill(box[:, None], 0.0)
        return x


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - m) / s


# --------------------------------------------------------------------------- ema
class GCELoss(nn.Module):
    """Generalized Cross-Entropy (Zhang & Sabuncu 2018) — noise-robust loss.

    L = sum_c t_c * (1 - p_c^q) / q, with optional label smoothing on the soft
    targets t. For q -> 0 it becomes label-smoothed CE. Per-sample reduction so it
    plugs into the existing weighted/mixup loss machinery unchanged.
    """

    def __init__(self, q: float = 0.7, label_smoothing: float = 0.0):
        super().__init__()
        self.q = q
        self.ls = label_smoothing

    def forward(self, logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        p = torch.softmax(logits.float(), dim=1)
        with torch.no_grad():
            t = torch.zeros_like(p)
            t.scatter_(1, target.unsqueeze(1), 1.0)
            if self.ls > 0:
                t = t * (1.0 - self.ls) + self.ls / p.shape[1]
        pc = p.clamp(1e-4, 1.0)
        return (t * (1.0 - pc.pow(self.q))).sum(1) / self.q


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, X_u8: torch.Tensor, size: int, mean, std, bs: int = 512, tta=True, want_ch=1):
    model.eval()
    n = X_u8.shape[0]
    out = torch.zeros(n, N_CLASSES, dtype=torch.float32, device=X_u8.device)
    variants = [(False, 0, 0)]
    if tta:
        variants += [(True, 0, 0)]
        for f in (False, True):
            variants += [(f, 0, 2), (f, 0, -2), (f, 2, 0), (f, -2, 0)]
    for flip, dy, dx in variants:
        for s in range(0, n, bs):
            xb = X_u8[s:s + bs]
            x = xb.float().div_(255.0)
            if dy or dx:
                x = F.pad(x, (2, 2, 2, 2), mode="replicate")
                x = x[:, :, 2 + dy:2 + dy + 48, 2 + dx:2 + dx + 48]
            if flip:
                x = torch.flip(x, dims=[3])
            if x.shape[-1] != size:
                x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            if want_ch == 3 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = normalize(x, mean, std)
            if x.is_cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", enabled=x.is_cuda, dtype=torch.float16):
                p = F.softmax(model(x), dim=1)
            out[s:s + bs] += p.float()
    return (out / len(variants)).cpu().numpy()


# --------------------------------------------------------------------------- train
def train_fold(args, Xtr_u8, ytr, tr_idx, va_idx, Xte_u8, device, pl=(None, None), fold=0):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    pl_X, pl_y = pl

    net, pretrained_ok, want_ch = build_model(args.arch, args.size, args.pretrained, width=args.width)
    net.to(device)
    if args.channels_last and device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
    if device.type == "cuda" and args.dp and torch.cuda.device_count() > 1:
        log(f"using DataParallel over {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)
    raw_net = net.module if isinstance(net, nn.DataParallel) else net

    if want_ch == 3:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5], [0.25]
    aug = (FastAug(args.size, strong=not args.weak_aug)
           if args.aug == "fast" else None)
    if aug is None:
        geo = make_aug(args.size, strong=not args.weak_aug)
        do_erase = not args.weak_aug

        def aug(x_u8):  # v2 fallback path, un-normalised [0,1] output
            x = geo(x_u8)
            return random_erasing(x) if do_erase else x

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, (len(tr_idx) + (len(pl[0]) if pl[0] is not None else 0)) // args.bs)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    if getattr(args, "loss", "ce") == "gce":
        crit = GCELoss(q=args.gce_q, label_smoothing=args.label_smooth)
        log(f"loss: GCE (q={args.gce_q}, ls={args.label_smooth})")
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(raw_net, args.ema_decay) if args.ema > 0 else None

    y_tr = torch.from_numpy(ytr[tr_idx]).to(device)
    X_tr = Xtr_u8[tr_idx]
    w_tr = torch.ones(len(tr_idx), device=device)
    if pl_X is not None and len(pl_X):
        X_tr = torch.cat([X_tr, pl_X], dim=0)
        y_tr = torch.cat([y_tr, pl_y], dim=0)
        w_tr = torch.cat([w_tr, torch.full((len(pl_X),), args.pl_weight, device=device)])
        log(f"pseudo-labels in fold {len(pl_X)} samples (loss weight {args.pl_weight}), "
            f"train set now {len(y_tr)}")
    step = 0
    t0 = time.time()
    hist = []
    best_val, best_raw, best_ema, bad_evals, stop_now = -1.0, None, None, 0, False
    for ep in range(args.epochs):
        net.train()
        if ema is not None and ep == args.ema_min_epoch - 1:
            ema = EMA(raw_net, args.ema_decay)  # restart from trained weights, not random init
            log("ema re-initialised from current weights")
        perm = torch.randperm(len(y_tr), device="cpu").numpy()
        run_loss, nb = 0.0, 0
        for bi in range(steps_per_epoch):
            sel = perm[bi * args.bs:(bi + 1) * args.bs]
            xb = X_tr[torch.from_numpy(sel)]
            yb = y_tr[torch.from_numpy(sel)]
            wb = w_tr[torch.from_numpy(sel)]
            with torch.no_grad():
                x = aug(xb)
                if want_ch == 3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                x = normalize(x, mean, std)
                if x.is_cuda and args.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            use_mix = args.mix_prob > 0 and random.random() < args.mix_prob
            if use_mix:
                lam = float(np.random.beta(args.mix_alpha, args.mix_alpha))
                lam = max(lam, 1 - lam)
                pidx = torch.randperm(x.shape[0], device=x.device)
                x = lam * x + (1 - lam) * x[pidx]
            with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                out = net(x)
                loss = (crit(out, yb) * wb).mean()
                if use_mix:
                    loss = lam * loss + (1 - lam) * (crit(out, yb[pidx]) * wb).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(raw_net)
            run_loss += float(loss.detach())
            nb += 1
            step += 1
        msg = f"epoch {ep + 1}/{args.epochs} loss={run_loss / max(nb,1):.4f} lr={lr_at(step):.2e} {time.time() - t0:.0f}s"
        heartbeat(args.out, stage="epoch", fold=fold, epoch=ep + 1, of=args.epochs,
                  loss=round(run_loss / max(nb, 1), 4), secs=int(time.time() - t0))
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            net.eval()
            m = net.module if isinstance(net, nn.DataParallel) else net
            raw_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
            va = torch.from_numpy(va_idx)
            with torch.no_grad():
                pv = predict(m, Xtr_u8[va], args.size, mean, std, bs=args.bs * 2, tta=False,
                             want_ch=want_ch)
            acc_raw = float((pv.argmax(1) == np.asarray(ytr)[va_idx]).mean())
            acc_ema = None
            if ema is not None and ep >= args.ema_min_epoch:
                m.load_state_dict({k: v.to(dtype=m.state_dict()[k].dtype) for k, v in ema.shadow.items()})
                with torch.no_grad():
                    pv2 = predict(m, Xtr_u8[va], args.size, mean, std, bs=args.bs * 2, tta=False,
                                  want_ch=want_ch)
                acc_ema = float((pv2.argmax(1) == np.asarray(ytr)[va_idx]).mean())
                m.load_state_dict(raw_sd)
            msg += f" | val_raw={acc_raw:.4f}" + (f" val_ema={acc_ema:.4f}" if acc_ema is not None else "")
            hist.append({"epoch": ep + 1, "loss": run_loss / max(nb, 1), "val_raw": acc_raw, "val_ema": acc_ema})
            # --- early stopping + best-checkpoint bookkeeping (off by default) ---
            if args.early_patience > 0:
                metric = acc_ema if acc_ema is not None else acc_raw
                if metric > best_val + 1e-4:
                    best_val = metric
                    best_raw = {k: v.detach().clone().float().cpu() for k, v in m.state_dict().items()}
                    best_ema = ({k: v.detach().clone() for k, v in ema.shadow.items()} if ema is not None else None)
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= args.early_patience:
                        stop_now = True
                        msg += f" | early-stop (best={best_val:.4f})"
        log(msg)
        if stop_now:
            m.load_state_dict({k: v.to(dtype=m.state_dict()[k].dtype) for k, v in best_raw.items()})
            if ema is not None and best_ema is not None:
                ema.shadow = best_ema
            log(f"early stopped at epoch {ep + 1}; restored best weights (val={best_val:.4f})")
            break

    m = net.module if isinstance(net, nn.DataParallel) else net
    raw_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    res = {}
    for tag in (["raw"] + (["ema"] if ema is not None else [])):
        if tag == "ema":
            m.load_state_dict({k: v.to(dtype=m.state_dict()[k].dtype) for k, v in ema.shadow.items()})
        oof = predict(m, Xtr_u8[va_idx], args.size, mean, std, bs=args.bs * 2, tta=args.tta,
                      want_ch=want_ch)
        test = predict(m, Xte_u8, args.size, mean, std, bs=args.bs * 2, tta=args.tta,
                       want_ch=want_ch)
        res[tag] = (oof, test)
        acc = float((oof.argmax(1) == np.asarray(ytr)[va_idx]).mean())
        log(f"fold done [{tag}] oof acc (tta={args.tta}) = {acc:.4f}")
        res[tag + "_acc"] = acc
    m.load_state_dict(raw_sd)
    if args.save_ckpt:
        torch.save(
            {"raw": {k: v.detach().cpu() for k, v in raw_sd.items()},
             "ema": ({k: v.detach().cpu() for k, v in ema.shadow.items()} if ema is not None else None),
             "arch": args.arch, "size": args.size, "want_ch": want_ch},
            os.path.join(args.out, f"fold{fold}_ckpt.pt"),
        )
    return res, hist, pretrained_ok


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="scratch48")
    p.add_argument("--width", type=int, default=64, help="stem width of the from-scratch net")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--pretrained", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--loss", default="ce", choices=["ce", "gce"], help="training loss (gce = noise-robust)")
    p.add_argument("--gce-q", type=float, default=0.7, help="GCE q parameter (0<q<=1)")
    p.add_argument("--early-patience", type=int, default=0,
                   help="early stop after N non-improving evals (0=off); restores best weights")
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--mix-alpha", type=float, default=0.2)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-min-epoch", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-list", default="")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--weak-aug", type=int, default=0)
    p.add_argument("--aug", default="fast", choices=["fast", "v2"])
    p.add_argument("--channels-last", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-minutes", type=float, default=1e9)
    p.add_argument("--fast", type=int, default=0)
    p.add_argument("--save-ckpt", type=int, default=0)
    p.add_argument("--allow-cpu", type=int, default=0)
    p.add_argument("--pl-file", default="", help="npy of test probabilities for pseudo-labelling")
    p.add_argument("--pl-b64", default="", help="base64-encoded float32 (n_test,6) array for pseudo-labelling")
    p.add_argument("--pl-thr", type=float, default=0.9)
    p.add_argument("--pl-weight", type=float, default=1.0, help="loss weight of pseudo-labelled samples")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__} cuda={torch.cuda.device_count()}")
    if device.type != "cuda" and not args.allow_cpu:
        log("FATAL: no CUDA device in this session - aborting instead of silently "
            "training on CPU for hours. Re-run later or pass --allow-cpu 1.")
        sys.exit(2)
    if device.type == "cuda":
        log("gpu:", torch.cuda.get_device_name(0))

    root = find_data_root()
    log("data root:", root)
    Xtr, ytr, Xte, names, per_class = load_split(root)
    log(f"train {Xtr.shape} test {Xte.shape} per_class {per_class}")
    heartbeat(args.out, stage="data_loaded", n_train=int(len(ytr)), n_test=int(len(Xte)))
    if args.fast:
        rng = np.random.RandomState(0)
        keep = []
        for c in range(N_CLASSES):
            idx = np.where(ytr == c)[0]
            rng.shuffle(idx)
            keep += list(idx[: max(30, len(idx) // 20)])
        keep = np.sort(np.array(keep))
        Xtr, ytr = Xtr[keep], ytr[keep]
        Xte = Xte[:400]
        names = names[:400]
        log(f"FAST mode: train {Xtr.shape} test {Xte.shape}")

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    folds = list(skf.split(np.zeros(len(ytr)), ytr))
    if args.fold_list:
        want = [int(x) for x in args.fold_list.split(",")]
    else:
        want = list(range(args.folds))

    os.makedirs(args.out, exist_ok=True)
    fold_of = np.full(len(ytr), -1, np.int64)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k
    np.save(os.path.join(args.out, "fold_ids.npy"), fold_of)

    Xtr_u8 = torch.from_numpy(Xtr).to(device)
    Xte_u8 = torch.from_numpy(Xte).to(device)
    pl = (None, None)
    if args.pl_b64 or args.pl_file:
        if args.pl_b64:
            probs = np.load(io.BytesIO(base64.b64decode(args.pl_b64)))
        else:
            probs = np.load(resolve_file(args.pl_file))
        if args.fast and len(probs) > len(Xte):
            probs = probs[: len(Xte)]
        assert len(probs) == len(Xte), (len(probs), len(Xte))
        keep = np.where(probs.max(1) >= args.pl_thr)[0]
        pl = (Xte_u8[torch.from_numpy(keep)], torch.from_numpy(probs[keep].argmax(1)).to(device))
        dist = np.bincount(probs[keep].argmax(1), minlength=N_CLASSES) / max(1, len(keep))
        log(f"pseudo-labelling: {len(keep)}/{len(probs)} test images with conf>={args.pl_thr}; "
            f"class mix " + ", ".join(f"{c}:{d:.2f}" for c, d in enumerate(dist)))
    run_name = os.path.basename(os.path.normpath(args.out))
    summary = {"args": vars(args), "folds_done": [], "per_fold": {}, "started": time.time()}
    done_probs = []
    for k in want:
        if (time.time() - t_start) / 60 > args.max_minutes:
            log("time budget exhausted, stopping")
            summary["stopped_early"] = True
            break
        tr_idx, va_idx = folds[k]
        log(f"===== fold {k}: train {len(tr_idx)} val {len(va_idx)} =====")
        res, hist, pre_ok = train_fold(args, Xtr_u8, ytr, tr_idx, va_idx, Xte_u8, device, pl=pl, fold=k)
        for tag in ("raw", "ema"):
            if tag in res:
                oof, test = res[tag]
                np.save(os.path.join(args.out, f"fold{k}_oof_{tag}.npy"), oof.astype(np.float32))
                np.save(os.path.join(args.out, f"fold{k}_test_{tag}.npy"), test.astype(np.float32))
        summary["folds_done"].append(k)
        summary["per_fold"][str(k)] = {"hist": hist,
                                       "oof_raw_acc": res.get("raw_acc"),
                                       "oof_ema_acc": res.get("ema_acc"),
                                       "pretrained_ok": pre_ok}
        use = "ema" if "ema_acc" in res and (res["ema_acc"] or 0) >= (res.get("raw_acc") or 0) else "raw"
        done_probs.append(res[use][1])
        log(f"fold {k} best variant {use} acc={res.get(use + '_acc'):.4f}")

    if done_probs:
        tm = np.mean(done_probs, axis=0)
        np.save(os.path.join(args.out, "test_mean.npy"), tm.astype(np.float32))
        with open(os.path.join(args.out, f"submission_{run_name}.csv"), "w", newline="") as f:
            f.write("ID,Emotion\n")
            for n, v in zip(names, tm.argmax(1)):
                f.write(f"{n},{int(v)}\n")
        log("wrote submission")

    summary["elapsed_min"] = (time.time() - t_start) / 60
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log("all done:", json.dumps({k: v for k, v in summary.items() if k in ("folds_done", "elapsed_min")}))


if __name__ == "__main__":
    main()
