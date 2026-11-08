import numpy as np
import cv2
from common import load_split, resize_to_512, dice
from eval_oof import remove_small

cv2.setNumThreads(8)


def load_probs():
    probs = {}
    for f in ['oof_best_03.npz', 'oof_best_4.npz']:
        d = np.load(f)
        for k in d.files:
            probs[int(k)] = d[k].astype(np.float32)
    return probs


def hysteresis(prob, hi, lo):
    seed = (prob > hi).astype(np.uint8)
    mask = (prob > lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    if n <= 1:
        return mask
    has = np.zeros(n, np.uint8)
    has[np.unique(lab[seed > 0])] = 1
    has[0] = 0
    return has[lab]


def comp_mean_filter(prob, thr, mean_thr):
    mask = (prob > thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    for k in range(1, n):
        sel = lab == k
        if prob[sel].mean() >= mean_thr:
            out[sel] = 1
    return out


def frac_threshold(prob, fov, target):
    """Threshold so that predicted fraction inside FOV equals target (percentile)."""
    vals = prob[fov > 0]
    if len(vals) == 0:
        return (prob > 0.5).astype(np.uint8)
    t = np.quantile(vals, 1.0 - target)
    return (prob > t).astype(np.uint8)


def main():
    _, labs, fovs = load_split(range(1, 21), 'train')
    probs = load_probs()
    ids = sorted(probs)
    gt = {i: resize_to_512(labs[i]) for i in ids}

    def ev(mask_fn, name, params):
        vals = []
        for i in ids:
            m = mask_fn(probs[i], fovs[i], *params)
            vals.append(dice(resize_to_512(m), gt[i]))
        print('%-42s mean %.4f' % (name, np.mean(vals)), flush=True)

    thr = 0.45
    ev(lambda p, f, t: (p > t).astype(np.uint8), 'baseline thr=%.2f' % thr, (thr,))
    print('--- hysteresis hi/lo ---')
    for hi in [0.5, 0.6, 0.7, 0.8]:
        for lo in [0.1, 0.2, 0.3, 0.4]:
            ev(lambda p, f, h, l: hysteresis(p, h, l), 'hysteresis hi=%.1f lo=%.1f' % (hi, lo), (hi, lo))
    print('--- component mean-prob filter ---')
    for cthr in [0.3, 0.4, 0.45]:
        for mthr in [0.1, 0.2, 0.3, 0.4]:
            ev(lambda p, f, t, m: comp_mean_filter(p, t, m),
               'comp thr=%.2f mean>=%.1f' % (cthr, mthr), (cthr, mthr))
    print('--- closing at full res ---')
    for k in [3, 5]:
        for it in [1, 2]:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            ev(lambda p, f, t, kk, ii: cv2.morphologyEx((p > t).astype(np.uint8), cv2.MORPH_CLOSE, kk, iterations=ii),
               'close k=%d it=%d thr=%.2f' % (k, it, thr), (thr, ker, it))
    print('--- per-image fraction matching (inside FOV) ---')
    for target in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]:
        ev(lambda p, f, t: frac_threshold(p, f, t), 'fraction target=%.2f' % target, (target,))


if __name__ == '__main__':
    main()
