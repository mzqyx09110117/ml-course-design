import os
import numpy as np
import cv2
from PIL import Image
from common import ROOT, CACHE, get_fov, make_input


def main():
    os.makedirs(CACHE, exist_ok=True)
    for split in ['train', 'test']:
        for i in range(1, 21):
            ip = os.path.join(ROOT, split, 'image', f'{i}.jpg')
            rgb = np.array(Image.open(ip).convert('RGB'))
            fov = get_fov(rgb)
            img = make_input(rgb, fov)
            if split == 'train':
                lp = os.path.join(ROOT, 'train', 'label', f'{i}.jpg')
                lab = (np.array(Image.open(lp).convert('L')) > 127).astype(np.uint8)
            else:
                lab = np.zeros(rgb.shape[:2], np.uint8)
            np.savez_compressed(os.path.join(CACHE, f'{split}_{i}.npz'), img=img, label=lab, fov=fov)
            print(split, i, img.shape, 'fov %.3f' % fov.mean(), 'vessel %.4f' % lab.mean())


if __name__ == '__main__':
    main()
