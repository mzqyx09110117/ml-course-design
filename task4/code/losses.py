import torch
import torch.nn.functional as F


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(logits, target, iters=8, smooth=1.0):
    p = torch.sigmoid(logits)
    skel_p = soft_skel(p, iters)
    skel_t = soft_skel(target, iters)
    tprec = (torch.sum(skel_p * target, dim=(1, 2, 3)) + smooth) / (torch.sum(skel_p, dim=(1, 2, 3)) + smooth)
    tsens = (torch.sum(skel_t * p, dim=(1, 2, 3)) + smooth) / (torch.sum(skel_t, dim=(1, 2, 3)) + smooth)
    cld = 1 - 2 * tprec * tsens / (tprec + tsens)
    return cld.mean()
