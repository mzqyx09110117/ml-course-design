"""EDA: label distribution, exact/near duplicate analysis between train and test.

Uses only the provided data (no external datasets).
"""
import os
import hashlib
import collections
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fer_data", "fer_data")
TRAIN, TEST = os.path.join(ROOT, "train"), os.path.join(ROOT, "test")
CLASSES = sorted(os.listdir(TRAIN))
print("classes:", CLASSES)

train_paths, train_labels = [], []
for ci, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(TRAIN, c))):
        train_paths.append(os.path.join(TRAIN, c, f))
        train_labels.append(ci)
test_paths = sorted(os.listdir(TEST))
test_paths = [os.path.join(TEST, f) for f in test_paths]
print("train", len(train_paths), "test", len(test_paths))


def load(p):
    im = Image.open(p).convert("L")
    if im.size != (48, 48):
        im = im.resize((48, 48))
    return np.asarray(im, dtype=np.uint8)


def dhash(a):
    d = a[:, 1:] > a[:, :-1]
    bits = d.flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def exhash(a):
    h = hashlib.blake2b(a.tobytes(), digest_size=16)
    return h.hexdigest()


Xtr = np.zeros((len(train_paths), 48, 48), np.uint8)
Xte = np.zeros((len(test_paths), 48, 48), np.uint8)
for i, p in enumerate(train_paths):
    Xtr[i] = load(p)
for i, p in enumerate(test_paths):
    Xte[i] = load(p)

tr_ex = collections.defaultdict(list)
for i in range(len(Xtr)):
    tr_ex[exhash(Xtr[i])].append(i)
te_ex = [exhash(Xte[i]) for i in range(len(Xte))]

# exact duplicates inside train
dup_groups = [v for v in tr_ex.values() if len(v) > 1]
n_dup_imgs = sum(len(v) for v in dup_groups)
inc = 0
for v in dup_groups:
    labs = [train_labels[i] for i in v]
    if len(set(labs)) > 1:
        inc += 1
print(f"train: {len(Xtr)} imgs, {len(tr_ex)} unique hashes, {len(dup_groups)} dup groups, "
      f"{n_dup_imgs} imgs in dup groups, {inc} conflicting-label groups")

# exact overlap train<->test
hit = sum(1 for h in te_ex if h in tr_ex)
print(f"test imgs with EXACT train twin: {hit}/{len(Xte)} = {hit/len(Xte):.4f}")

# label consistency for test images with exact twins
agree = collections.Counter()
for i, h in enumerate(te_ex):
    if h in tr_ex:
        labs = {train_labels[j] for j in tr_ex[h]}
        agree["consistent" if len(labs) == 1 else "conflict"] += 1
print("exact-twin label consistency:", dict(agree))

# near-dup via dhash, test vs train
Htr = np.array([dhash(Xtr[i]) for i in range(len(Xtr))], dtype=np.uint64)
Hte = np.array([dhash(Xte[i]) for i in range(len(Xte))], dtype=np.uint64)
print("computing near-dup distances (test x train)...")
for thr in (0, 2, 4, 6):
    n_hit = 0
    for i in range(len(Hte)):
        x = Htr ^ Hte[i]
        # popcount
        d = np.zeros(len(x), np.uint16)
        t = x.copy()
        while t.any():
            d += (t & np.uint64(1)).astype(np.uint16)
            t = t >> np.uint64(1)
        if d.min() <= thr:
            n_hit += 1
    print(f"dhash dist<={thr}: test imgs with train neighbor: {n_hit} ({n_hit/len(Hte):.4f})")

np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "Xtr.npy"), Xtr)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "Xte.npy"), Xte)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "meta.txt"), "w") as f:
    f.write("classes " + ",".join(CLASSES) + "\n")
    for i, c in enumerate(CLASSES):
        f.write(f"{c} {sum(1 for l in train_labels if l==i)}\n")
print("saved arrays")
