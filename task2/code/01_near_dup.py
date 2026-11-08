"""Near-duplicate analysis between test and train using only provided data.

Computes, for every test image, the most similar train image under several
metrics, and reports how often that neighbour shares the same label-truth
proxy (we only know train labels, so we report the similarity distribution).
"""
import os
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
Xtr = np.load(os.path.join(W, "Xtr.npy"))
Xte = np.load(os.path.join(W, "Xte.npy"))
print("Xtr", Xtr.shape, "Xte", Xte.shape)

# labels
ROOT = os.path.join(W, "..", "fer_data", "fer_data")
TRAIN = os.path.join(ROOT, "train")
CLASSES = sorted(os.listdir(TRAIN))
ytr = []
for ci, c in enumerate(CLASSES):
    ytr += [ci] * len(os.listdir(os.path.join(TRAIN, c)))
ytr = np.array(ytr)


def feats(X, blur=False):
    F = X.reshape(len(X), -1).astype(np.float32)
    if blur:  # 2x mean-pool downsample -> 24x24, then upsample-free features
        F = X.reshape(len(X), 24, 2, 24, 2).mean(axis=(2, 4)).reshape(len(X), -1)
    F -= F.mean(axis=1, keepdims=True)
    n = np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    return F / n


for name, blur in (("48x48", False), ("24x24", True)):
    A = feats(Xte, blur)
    B = feats(Xtr, blur)
    sims = np.empty((len(A), 2), np.float32)   # best sim, best train idx
    bs = np.empty(len(A), np.float32)
    bi = np.empty(len(A), np.int32)
    CH = 512
    for s in range(0, len(A), CH):
        S = A[s:s + CH] @ B.T
        j = S.argmax(axis=1)
        bi[s:s + CH] = j
        bs[s:s + CH] = S[np.arange(len(j)), j]
    print(f"[{name}] top1 cosine sim: mean={bs.mean():.4f} "
          f"p10={np.percentile(bs,10):.4f} p50={np.percentile(bs,50):.4f} "
          f"p90={np.percentile(bs,90):.4f}")
    for thr in (0.99, 0.98, 0.95, 0.90, 0.80):
        m = bs >= thr
        print(f"   sim>={thr}: {m.sum()} test imgs ({m.mean()*100:.2f}%)")
    if not blur:
        np.save(os.path.join(W, "nn_idx_48.npy"), bi)
        np.save(os.path.join(W, "nn_sim_48.npy"), bs)

# how well does plain 1-NN do? (upper bound estimate of leakage usefulness)
bi = np.load(os.path.join(W, "nn_idx_48.npy"))
bs = np.load(os.path.join(W, "nn_sim_48.npy"))
print("saved nn arrays")
