"""Build compact numpy arrays (images + labels) from the provided FER folders."""
import os
import numpy as np
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(W, "..", "fer_data", "fer_data")
TRAIN, TEST = os.path.join(ROOT, "train"), os.path.join(ROOT, "test")
# official label ids from the competition rules
CLASS_ORDER = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

assert sorted(os.listdir(TRAIN)) == sorted(CLASS_ORDER), os.listdir(TRAIN)
train_paths, ytr = [], []
for ci, c in enumerate(CLASS_ORDER):
    for f in sorted(os.listdir(os.path.join(TRAIN, c))):
        train_paths.append(os.path.join(TRAIN, c, f))
        ytr.append(ci)
test_names = sorted(os.listdir(TEST))
test_paths = [os.path.join(TEST, f) for f in test_names]


def load(p):
    im = Image.open(p).convert("L")
    return np.asarray(im, dtype=np.uint8)


Xtr = np.stack([load(p) for p in train_paths])
Xte = np.stack([load(p) for p in test_paths])
ytr = np.array(ytr, np.int64)

np.save(os.path.join(W, "Xtr.npy"), Xtr)
np.save(os.path.join(W, "Xte.npy"), Xte)
np.save(os.path.join(W, "ytr.npy"), ytr)
with open(os.path.join(W, "test_names.txt"), "w") as f:
    f.write("\n".join(test_names))
print("saved", Xtr.shape, Xte.shape, ytr.shape, "classes:", CLASS_ORDER)
print("per-class counts:", np.bincount(ytr, minlength=6).tolist())
