import os, math, time
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DATA = '/home/mzqyx09110117/桌面/test3/detection'
IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- model
def conv_bn(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FoveaNet(nn.Module):
    """ResNet-FPN heatmap regressor, output stride 4."""

    def __init__(self, backbone='resnet18', pretrained=True, freeze_stem=True, width=96, in_ch=3, stride2=False):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        m = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        if in_ch != 3:
            old = self.stem[0]
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride,
                            old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                if in_ch > 3:
                    new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch - 3, 1, 1)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            self.stem[0] = new
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.frozen = [self.stem, self.layer1] if freeze_stem else []
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad_(False)
            for p in self.layer1.parameters():
                p.requires_grad_(False)
        ch = [64, 128, 256, 512] if backbone in ('resnet18', 'resnet34') else [256, 512, 1024, 2048]
        self.lat4 = nn.Conv2d(ch[3], width, 1)
        self.lat3 = nn.Conv2d(ch[2], width, 1)
        self.lat2 = nn.Conv2d(ch[1], width, 1)
        self.lat1 = nn.Conv2d(ch[0], width, 1)
        self.smooth = conv_bn(width, width)
        self.head = nn.Conv2d(width, 1, 1)
        self.stride2 = stride2
        if stride2:
            self.smooth2 = conv_bn(width, width // 2)
            self.head2 = nn.Conv2d(width // 2, 1, 1)
        nn.init.constant_(self.head.bias, -2.0)

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:  # keep frozen BN statistics
            m.eval()
        return self

    def forward(self, x):
        c1 = self.layer1(self.stem(x))       # /4
        c2 = self.layer2(c1)                 # /8
        c3 = self.layer3(c2)                 # /16
        c4 = self.layer4(c3)                 # /32
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        out = self.smooth(p1)
        if self.stride2:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            return self.head2(self.smooth2(out))
        return self.head(out)


def local_softargmax(hm, cx, cy, r=8, gamma=2.0, eps=1e-6):
    """Weighted centroid of heatmap inside a window centred on (cx,cy) (cell coords)."""
    B, _, H, W = hm.shape
    ys = torch.arange(H, device=hm.device, dtype=hm.dtype).view(1, H, 1)
    xs = torch.arange(W, device=hm.device, dtype=hm.dtype).view(1, 1, W)
    d2 = (xs - cx.view(B, 1, 1)) ** 2 + (ys - cy.view(B, 1, 1)) ** 2
    mask = (d2 <= r * r).to(hm.dtype)
    w = torch.relu(hm).pow(gamma) * mask + eps * mask
    w = w / w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    px = (w * xs).sum(dim=(2, 3))
    py = (w * ys).sum(dim=(2, 3))
    return px, py


def heatmap_peak(hm, r=6, gamma=2.0):
    """Inference: global argmax then local weighted centroid. hm: 1x1xHxW."""
    hm = hm[0, 0]
    idx = int(torch.argmax(hm))
    py, px = divmod(idx, hm.shape[1])
    H, W = hm.shape
    y0, y1 = max(0, py - r), min(H, py + r + 1)
    x0, x1 = max(0, px - r), min(W, px + r + 1)
    win = torch.relu(hm[y0:y1, x0:x1]).pow(gamma)
    ys = torch.arange(y0, y1, device=hm.device, dtype=hm.dtype).view(-1, 1)
    xs = torch.arange(x0, x1, device=hm.device, dtype=hm.dtype).view(1, -1)
    s = win.sum().clamp_min(1e-12)
    return float((win * xs).sum() / s), float((win * ys).sum() / s)


# ---------------------------------------------------------------- data
def load_all_images(size=1024):
    """Pre-resize all train+test images to square size x size uint8."""
    imgs, names, sizes = {}, [], {}
    for split in ('train', 'test'):
        for f in sorted(os.listdir(f'{DATA}/{split}')):
            if not f.lower().endswith('.jpg'):
                continue
            n = f.split('.')[0]
            im = cv2.imread(f'{DATA}/{split}/{f}')
            H, W = im.shape[:2]
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            imgs[n] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            names.append(n)
            sizes[n] = (W, H)
    return imgs, names, sizes


def load_gt():
    df = pd.read_csv(f'{DATA}/fovea_localization_train_GT.csv', dtype={'data': str})
    gt = {}
    for _, r in df.iterrows():
        gt[str(int(r.data)).zfill(4)] = (float(r.Fovea_X), float(r.Fovea_Y))
    return gt


def rand_affine(out=512, inp=1024, scale_rng=(0.85, 1.25), rot_rng=(-15, 15),
                trans=0.07, flip_p=0.5, rot_max=None):
    scale = np.random.uniform(*scale_rng)
    rot = np.random.uniform(*rot_rng)
    tx = np.random.uniform(-trans, trans) * out
    ty = np.random.uniform(-trans, trans) * out
    flip = np.random.rand() < flip_p
    s = scale * out / inp
    a = math.radians(rot)
    S = s * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float64)
    A = S.copy()
    if flip:
        A[0, :] *= -1
    c_in = inp / 2.0
    c_out = out / 2.0
    b = np.array([c_out + tx, c_out + ty]) - A @ np.array([c_in, c_in])
    M = np.hstack([A, b[:, None]])
    return M


def apply_affine(img, M, out=512):
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply_clahe(img):
    """RGB uint8 -> CLAHE-enhanced RGB uint8."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def add_haze(img):
    """img float32 HxWx3 in [0,1]; add a random soft bright blob (haze/reflection)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
    sigma = np.random.uniform(80, 220)
    amp = np.random.uniform(0.15, 0.45)
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)
    return np.clip(img + amp * g[..., None], 0, 1)


def color_aug(img):
    """img float32 HxWx3 in [0,255]."""
    x = img.astype(np.float32) / 255.0
    if np.random.rand() < 0.8:
        alpha = np.random.uniform(0.85, 1.15)   # contrast
        beta = np.random.uniform(-0.08, 0.08)   # brightness
        x = x * alpha + beta
    if np.random.rand() < 0.5:
        gamma = np.random.uniform(0.8, 1.25)
        x = np.clip(x, 0, 1) ** gamma
    if np.random.rand() < 0.5:
        gain = np.random.uniform(0.95, 1.05, 3).astype(np.float32)
        x = x * gain
    if np.random.rand() < 0.15:
        x = cv2.GaussianBlur(x, (0, 0), np.random.uniform(0.5, 1.5))
    if np.random.rand() < 0.1:
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01
    return np.clip(x, 0, 1)


def normalize(x):
    x = (x - IMG_MEAN) / IMG_STD
    return x.transpose(2, 0, 1).astype(np.float32)


def make_heatmap(hm_size, ux, uy, sigma=1.5, radius=6):
    """ux,uy normalised [0,1] -> Gaussian heatmap hm_size x hm_size."""
    t = np.zeros((hm_size, hm_size), np.float32)
    cx, cy = ux * hm_size - 0.5, uy * hm_size - 0.5
    x0, x1 = max(0, int(cx) - radius), min(hm_size, int(cx) + radius + 2)
    y0, y1 = max(0, int(cy) - radius), min(hm_size, int(cy) + radius + 2)
    if x1 <= x0 or y1 <= y0:
        return t
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    gx = np.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ys - cy) ** 2) / (2 * sigma ** 2))
    t[y0:y1, x0:x1] = np.outer(gy, gx)
    return t


def disc_channel(cx, cy, size, sigma=10.0):
    ch = np.zeros((size, size), np.float32)
    if cx is None or not (-0.2 * size <= cx < 1.2 * size and -0.2 * size <= cy < 1.2 * size):
        return ch
    r = int(3 * sigma)
    x0, x1 = max(0, int(cx) - r), min(size, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(size, int(cy) + r + 1)
    xs = np.arange(x0, x1, dtype=np.float32); ys = np.arange(y0, y1, dtype=np.float32)
    gx = np.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ys - cy) ** 2) / (2 * sigma ** 2))
    ch[y0:y1, x0:x1] = np.outer(gy, gx)
    return ch


class FoveaDataset(torch.utils.data.Dataset):
    def __init__(self, names, imgs, sizes, gt, out=512, inp=1024, hm_size=128,
                 train=True, sigma=1.5, clahe_p=0.0, disc=None, offset_mode=False, haze_p=0.0):
        self.names, self.imgs, self.sizes, self.gt = names, imgs, sizes, gt
        self.out, self.inp, self.hm_size, self.train, self.sigma = out, inp, hm_size, train, sigma
        self.clahe_p = clahe_p
        self.disc = disc
        self.offset_mode = offset_mode
        self.haze_p = haze_p

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = self.imgs[n]
        W, H = self.sizes[n]
        x, y = self.gt[n]
        gx, gy = x / W * self.inp, y / H * self.inp   # in 1024 space
        if self.train:
            for _ in range(5):
                M = rand_affine(self.out, self.inp)
                px = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
                py = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]
                if 0.02 * self.out < px < 0.98 * self.out and 0.02 * self.out < py < 0.98 * self.out:
                    break
            warped = apply_affine(img, M, self.out)
            if self.disc is not None:
                dd = self.disc.get(n, (0.5, 0.5))
                dout = M @ np.array([dd[0] * self.inp, dd[1] * self.inp, 1.0])
                dcx, dcy = dout[0], dout[1]
            else:
                dcx = dcy = None
            if self.clahe_p > 0 and np.random.rand() < self.clahe_p:
                warped = apply_clahe(warped)
            warped = color_aug(warped)
            if self.haze_p > 0 and np.random.rand() < self.haze_p:
                warped = add_haze(warped)
            ux, uy = (px + 0.5) / self.out, (py + 0.5) / self.out
        else:
            warped = cv2.resize(img, (self.out, self.out), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            # inp == 2*out: gx in 1024-space maps to gx*out/inp in output pixels
            ux = (gx * self.out / self.inp + 0.5) / self.out
            uy = (gy * self.out / self.inp + 0.5) / self.out
            if self.disc is not None:
                dd = self.disc.get(n, (0.5, 0.5))
                dcx, dcy = dd[0] * self.out, dd[1] * self.out
            else:
                dcx = dcy = None
        if self.offset_mode and self.disc is not None:
            # target = fovea offset from the disc centre, mapped to [0,1]
            fpx = ux * self.out - 0.5
            fpy = uy * self.out - 0.5
            ux = (fpx - dcx) / self.out + 0.5
            uy = (fpy - dcy) / self.out + 0.5
        hm = make_heatmap(self.hm_size, ux, uy, self.sigma)
        xin = normalize(warped)
        if self.disc is not None:
            xin = np.concatenate([xin, disc_channel(dcx, dcy, self.out)[None]], 0)
        return (torch.from_numpy(xin), torch.from_numpy(hm).unsqueeze(0),
                torch.tensor([ux, uy], dtype=torch.float32))
