import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WORK, 'cache')


def get_fov(rgb):
    """Circular field-of-view mask for a fundus image."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (g > 10).astype(np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)
    return m


def make_input(rgb, fov):
    """5 channels: CLAHE(G), CLAHE(R), CLAHE(B), raw G, FOV; standardized inside FOV."""
    h, w = rgb.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    chans = []
    for c in range(3):
        chans.append(clahe.apply(rgb[:, :, c]).astype(np.float32) / 255.0)
    chans.append(rgb[:, :, 1].astype(np.float32) / 255.0)
    f = fov.astype(np.float32)
    sel = fov > 0
    for i in range(4):
        x = chans[i]
        mu = x[sel].mean()
        sd = x[sel].std() + 1e-6
        chans[i] = (x - mu) / sd
    out = np.stack(chans + [f], axis=0).astype(np.float16)
    return out


def load_split(ids, split, cache=None):
    """Load cached per-image arrays: img (C,H,W) float16, label (H,W) uint8, fov."""
    cache = cache or CACHE
    imgs, labs, fovs = {}, {}, {}
    for i in ids:
        p = os.path.join(cache, f'{split}_{i}.npz')
        d = np.load(p)
        imgs[i] = d['img']
        labs[i] = d['label']
        fovs[i] = d['fov']
    return imgs, labs, fovs


# ---------------- model ----------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=5, base=32, depth=4):
        super().__init__()
        cs = [base * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList([DoubleConv(in_ch if i == 0 else cs[i - 1], cs[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck_drop = nn.Dropout2d(0.1)
        self.up = nn.ModuleList([nn.ConvTranspose2d(cs[i], cs[i - 1], 2, 2) for i in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([DoubleConv(2 * cs[i - 1], cs[i - 1]) for i in range(depth - 1, 0, -1)])
        self.out = nn.Conv2d(cs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        x = self.bottleneck_drop(x)
        for j, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            x = torch.cat([x, skips[-1 - j]], 1)
            x = d(x)
        return self.out(x)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.r1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.r1(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.r2(y + self.skip(x))


class SCSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(1, c // r), 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(max(1, c // r), c, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class UNetV2(nn.Module):
    """Residual U-Net with scSE attention in the decoder."""

    def __init__(self, in_ch=5, base=32, depth=4):
        super().__init__()
        cs = [base * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList([ResBlock(in_ch if i == 0 else cs[i - 1], cs[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck_drop = nn.Dropout2d(0.1)
        self.up = nn.ModuleList([nn.ConvTranspose2d(cs[i], cs[i - 1], 2, 2) for i in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([nn.Sequential(ResBlock(2 * cs[i - 1], cs[i - 1]), SCSE(cs[i - 1]))
                                  for i in range(depth - 1, 0, -1)])
        self.out = nn.Conv2d(cs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        x = self.bottleneck_drop(x)
        for j, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            x = torch.cat([x, skips[-1 - j]], 1)
            x = d(x)
        return self.out(x)


class MTLUNet(UNetV2):
    """UNetV2 with auxiliary boundary + centerline heads (multi-task)."""

    def __init__(self, in_ch=5, base=32, depth=4):
        super().__init__(in_ch, base, depth)
        self.aux = nn.Sequential(nn.Conv2d(base, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, 1))

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        x = self.bottleneck_drop(x)
        for j, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            x = torch.cat([x, skips[-1 - j]], 1)
            x = d(x)
        return self.out(x), self.aux(x)


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (p * target).sum(dim=dims)
    fp = (p * (1 - target)).sum(dim=dims)
    fn = ((1 - p) * target).sum(dim=dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - ti.mean()


def dice_loss(logits, target, weight=None, eps=1.0):
    p = torch.sigmoid(logits)
    if weight is None:
        num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
        den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    else:
        num = 2 * (p * target * weight).sum(dim=(1, 2, 3)) + eps
        den = (p * weight).sum(dim=(1, 2, 3)) + (target * weight).sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def bce_dice_loss(logits, target, weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(1.5), reduction='none')
    if weight is None:
        bce = bce.mean()
    else:
        bce = (bce * weight).sum() / (weight.sum() + 1e-6)
    return bce + dice_loss(logits, target, weight)


# ---------------- inference ----------------
def _forward_padded(model, x):
    # x: (1,C,H,W) tensor
    _, _, h, w = x.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = torch.sigmoid(y)
    return y[:, :, :h, :w]


@torch.no_grad()
def predict_full(model, img, tta=True):
    """img: (5,H,W) float32 numpy. Returns probability map (H,W) float32."""
    model.eval()
    h, w = img.shape[1:]
    s = max(h, w)
    s = int(np.ceil(s / 16) * 16)
    x0 = np.zeros((img.shape[0], s, s), np.float32)
    x0[:, :h, :w] = img
    acc = np.zeros((s, s), np.float32)
    n = 0
    if not tta:
        variants = [(0, False)]
    elif tta == 4:
        variants = [(0, False), (2, False), (0, True), (2, True)]
    else:
        variants = [(k, f) for k in range(4) for f in (False, True)]
    for k, flip in variants:
        x = np.rot90(x0, k, axes=(1, 2))
        if flip:
            x = x[:, :, ::-1]
        xt = torch.from_numpy(np.ascontiguousarray(x))[None]
        y = _forward_padded(model, xt)[0, 0].numpy()
        if flip:
            y = y[:, ::-1]
        y = np.rot90(y, -k)
        acc += y
        n += 1
    prob = acc / n
    return prob[:h, :w]


def adapt_bn(model, imgs):
    """Recompute BatchNorm running statistics on unlabeled target images."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.train()
        m.reset_running_stats()
        m.momentum = None
    with torch.no_grad():
        for img in imgs:
            h, w = img.shape[1:]
            ph = (16 - h % 16) % 16
            pw = (16 - w % 16) % 16
            x = torch.from_numpy(np.ascontiguousarray(img))[None]
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode='replicate')
            model(x)
    for m in bns:
        m.momentum = 0.1
    model.eval()


def predict_full_ms(model, img, tta=True, scales=(1.0,)):
    """Multi-scale average of predict_full."""
    h, w = img.shape[1:]
    acc = np.zeros((h, w), np.float32)
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            im = img
        else:
            im = np.stack([cv2.resize(img[c], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                           for c in range(img.shape[0])])
        p = predict_full(model, im, tta=tta)
        if abs(s - 1.0) > 1e-6:
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        acc += p
    return acc / len(scales)


def resize_to_512(mask01):
    """Replicate the organizers' pipeline: PIL bilinear to 512 then >0."""
    from PIL import Image
    m = (mask01.astype(np.uint8) * 255)
    im = Image.fromarray(m, mode='L').resize((512, 512), Image.Resampling.BILINEAR)
    a = np.asarray(im)
    return (a > 0).astype(np.uint8)


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return 2.0 * (a & b).sum() / s
