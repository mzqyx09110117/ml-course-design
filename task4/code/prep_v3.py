import os
import numpy as np
import cv2
from PIL import Image
from common import ROOT, WORK, get_fov

CACHE_V3 = os.path.join(WORK, 'cache_v3')


def make_input_v3(rgb, fov):
    """8 channels: CLAHE(G), CLAHE(R), CLAHE(B), raw G, black-top-hat(G), CLAHE(L),
    normalized green G/(R+G+B), FOV. Standardized inside FOV."""
    rgbf = rgb.astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(rgb[:, :, 1])
    r_clahe = clahe.apply(rgb[:, :, 0])
    b_clahe = clahe.apply(rgb[:, :, 2])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_clahe = clahe.apply(lab[:, :, 0])
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, ker)
    denom = rgbf.sum(axis=2) + 1e-3
    norm_g = rgbf[:, :, 1] / denom
    chans = [g_clahe.astype(np.float32) / 255.0, r_clahe.astype(np.float32) / 255.0,
             b_clahe.astype(np.float32) / 255.0, rgbf[:, :, 1] / 255.0,
             tophat.astype(np.float32) / 255.0, l_clahe.astype(np.float32) / 255.0,
             norm_g.astype(np.float32)]
    sel = fov > 0
    for i in range(len(chans)):
        x = chans[i]
        mu = x[sel].mean()
        sd = x[sel].std() + 1e-6
        chans[i] = (x - mu) / sd
    return np.stack(chans + [fov.astype(np.float32)], axis=0).astype(np.float16)


def main():
    os.makedirs(CACHE_V3, exist_ok=True)
    for split in ['train', 'test']:
        for i in range(1, 21):
            ip = os.path.join(ROOT, split, 'image', f'{i}.jpg')
            rgb = np.array(Image.open(ip).convert('RGB'))
            fov = get_fov(rgb)
            img = make_input_v3(rgb, fov)
            if split == 'train':
                lp = os.path.join(ROOT, 'train', 'label', f'{i}.jpg')
                lab = (np.array(Image.open(lp).convert('L')) > 127).astype(np.uint8)
            else:
                lab = np.zeros(rgb.shape[:2], np.uint8)
            np.savez_compressed(os.path.join(CACHE_V3, f'{split}_{i}.npz'), img=img, label=lab, fov=fov)
            print(split, i, img.shape, flush=True)


if __name__ == '__main__':
    main()
