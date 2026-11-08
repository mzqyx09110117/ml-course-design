import os, math
import numpy as np
import cv2
import torch
from fovea_lib import DATA, normalize, make_heatmap, color_aug


def load_native_images(names=None):
    """Load original-resolution RGB images into memory (uint8)."""
    imgs, sizes = {}, {}
    for split in ('train', 'test'):
        for f in sorted(os.listdir(f'{DATA}/{split}')):
            if not f.lower().endswith('.jpg'):
                continue
            n = f.split('.')[0]
            if names is not None and n not in names:
                continue
            im = cv2.imread(f'{DATA}/{split}/{f}')
            imgs[n] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            sizes[n] = (im.shape[1], im.shape[0])
    return imgs, sizes


def crop_affine(cx, cy, S=512, scale=1.0, rot=0.0, flip=False):
    """Affine mapping input-image coords -> crop-output coords.
    Output is centred on (cx,cy) with zoom `scale` (1 = native res)."""
    a = math.radians(rot)
    R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]], np.float64)  # R(-rot)
    A = R * (1.0 / scale)
    if flip:
        A[0, :] *= -1
    c = np.array([cx, cy], np.float64)
    b = np.array([S / 2.0, S / 2.0]) - A @ c
    return np.hstack([A, b[:, None]])


class Stage2Dataset(torch.utils.data.Dataset):
    def __init__(self, names, imgs, sizes, gt, S=512, hm=128, sigma=1.2,
                 train=True, jitter=64.0, scale_rng=(0.9, 1.15), rot_rng=(-10, 10)):
        self.names, self.imgs, self.sizes, self.gt = names, imgs, sizes, gt
        self.S, self.hm, self.sigma = S, hm, sigma
        self.train, self.jitter = train, jitter
        self.scale_rng, self.rot_rng = scale_rng, rot_rng

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = self.imgs[n]
        gx, gy = self.gt[n]
        if self.train:
            cx = gx + np.random.uniform(-self.jitter, self.jitter)
            cy = gy + np.random.uniform(-self.jitter, self.jitter)
            sc = np.random.uniform(*self.scale_rng)
            rot = np.random.uniform(*self.rot_rng)
            flip = np.random.rand() < 0.5
        else:
            cx, cy, sc, rot, flip = gx, gy, 1.0, 0.0, False
        M = crop_affine(cx, cy, self.S, sc, rot, flip)
        crop = cv2.warpAffine(img, M, (self.S, self.S), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = M @ np.array([gx, gy, 1.0])
        ux = (out[0] + 0.5) / self.S
        uy = (out[1] + 0.5) / self.S
        if self.train:
            crop = color_aug(crop)
        else:
            crop = crop.astype(np.float32) / 255.0
        hm = make_heatmap(self.hm, ux, uy, self.sigma)
        return (torch.from_numpy(normalize(crop)), torch.from_numpy(hm).unsqueeze(0),
                torch.tensor([ux, uy], dtype=torch.float32))


def stage2_predict(model, img, cx, cy, S=512, tta_flip=True, hm=128, scale=1.0):
    """img native RGB; return refined (x,y) in native coords."""
    preds = []
    for fl in ([False, True] if tta_flip else [False]):
        M = crop_affine(cx, cy, S, scale=scale, flip=fl)
        crop = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        x = torch.from_numpy(normalize(crop.astype(np.float32) / 255.0)).unsqueeze(0)
        with torch.no_grad():
            o = model(x)
            p = torch.softmax(o.flatten(1), dim=1).view_as(o)
            from fovea_lib import heatmap_peak
            px, py = heatmap_peak(p)
        ux = (px + 0.5) / hm
        uy = (py + 0.5) / hm
        # invert: p_out = A p_in + b  ->  p_in = A^-1 (p_out - b); handles flip/rot/scale
        A = M[:, :2]
        b = M[:, 2]
        p_out = np.array([ux * S - 0.5, uy * S - 0.5])
        p_in = np.linalg.solve(A, p_out - b)
        preds.append(p_in)
    return float(np.mean([p[0] for p in preds])), float(np.mean([p[1] for p in preds]))
