#!/usr/bin/env python3
"""Blend run outputs into a submission and report OOF accuracy.

Usage:
  python work/ensemble.py --runs A=work/kaggle/A/output/out_A B=work/kaggle/B/output/out_B \
      [--dup 1] [--out submission_ens.csv]

Each run directory must contain fold{k}_oof_{raw,ema}.npy, fold{k}_test_{raw,ema}.npy and
fold_ids.npy (all written by fer_train.py).

Steps
  1. individual member OOF accuracy
  2. greedy ensemble selection with replacement (Caruana) over linear / log space
  3. optional per-class logit bias calibration (coordinate ascent on OOF accuracy)
  4. optional near-duplicate override / blend, tuned on OOF
  5. writes the submission, the report, and the blended test probabilities
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_CLASSES = 6


def load_run(path: str):
    fold_ids = np.load(os.path.join(path, "fold_ids.npy"))
    variants = {}
    for f in sorted(os.listdir(path)):
        if f.startswith("fold") and f.endswith("_oof_raw.npy"):
            k = int(f[4:f.index("_")])
            for variant in ("raw", "ema"):
                oo = os.path.join(path, f"fold{k}_oof_{variant}.npy")
                tt = os.path.join(path, f"fold{k}_test_{variant}.npy")
                if os.path.exists(oo) and os.path.exists(tt):
                    d = variants.setdefault(variant, {"oof": {}, "test": []})
                    d["oof"][k] = np.load(oo)
                    d["test"].append(np.load(tt))
    out = {}
    n = len(fold_ids)
    for v, d in variants.items():
        oof = np.full((n, N_CLASSES), np.nan, np.float32)
        for k, arr in d["oof"].items():
            oof[fold_ids == k] = arr
        out[v] = (oof, np.mean(d["test"], axis=0))
    return out, fold_ids


def acc_of(p: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(p[:, 0])
    return float((p[m].argmax(1) == y[m]).mean()) if m.sum() else float("nan")


def to_space(p: np.ndarray, space: str) -> np.ndarray:
    if space == "log":
        return np.log(np.clip(p, 1e-7, 1.0))
    return p


def probs_from(v: np.ndarray, space: str) -> np.ndarray:
    """Turn a member-space vector back into a probability distribution."""
    if space == "log":
        v = v - v.max(1, keepdims=True)
        e = np.exp(v)
        return e / e.sum(1, keepdims=True)
    v = np.clip(v, 0.0, None)
    s = v.sum(1, keepdims=True)
    return np.where(s > 0, v / np.maximum(s, 1e-12), 1.0 / N_CLASSES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--y", default=os.path.join(HERE, "ytr.npy"))
    ap.add_argument("--names", default=os.path.join(HERE, "test_names.txt"))
    ap.add_argument("--out", default="submission_ens.csv")
    ap.add_argument("--report", default=os.path.join(HERE, "ens_report.txt"))
    ap.add_argument("--dup", type=int, default=1)
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--save-probs", default=os.path.join(HERE, "test_probs_ens.npy"))
    ap.add_argument("--save-pl", default=os.path.join(HERE, "pl_probs.npy"))
    ap.add_argument("--steps", type=int, default=40, help="greedy selection steps")
    ap.add_argument("--refine", type=int, default=1, help="continuous weight hill-climbing after greedy")
    ap.add_argument("--stack", type=int, default=1, help="compare a logistic-regression stacker")
    ap.add_argument("--spec", default="", help="dir of a triangle-class (0/1/3) specialist run")
    ap.add_argument("--spec-variant", default="ema", help="which specialist variant to use (raw/ema)")
    a = ap.parse_args()

    y = np.load(a.y)
    rep = []

    def say(s):
        print(s, flush=True)
        rep.append(str(s))

    members = {}
    fold_ids = None
    for spec in a.runs:
        name, path = spec.split("=", 1)
        vs, fid = load_run(path)
        if fold_ids is None:
            fold_ids = fid
        elif not np.array_equal(fold_ids, fid):
            say(f"WARNING: fold ids of {name} differ from the first run")
        for v, (oof, test) in vs.items():
            members[(name, v)] = (oof, test)

    say("=== individual members (OOF acc) ===")
    singles = {k: acc_of(v[0], y) for k, v in members.items()}
    for k in sorted(singles, key=lambda k: -singles[k]):
        say(f"  {k[0]:>6s}/{k[1]:<4s} {singles[k]:.4f}")

    best_space, best_res = None, None
    for space in ("prob", "log"):
        ms = {k: (to_space(v[0], space), to_space(v[1], space)) for k, v in members.items()}
        counts = {}
        cur = None
        cur_acc = -1.0
        for it in range(a.steps):
            best_k, best_acc = None, -1.0
            for k in ms:
                trial = ms[k][0] if cur is None else cur + ms[k][0]
                with np.errstate(invalid="ignore"):
                    m = ~np.isnan(trial[:, 0])
                    acc = float((np.nanargmax(np.where(np.isnan(trial), -1e9, trial), axis=1)[m] == y[m]).mean())
                if acc > best_acc:
                    best_acc, best_k = acc, k
            if best_k is None:
                break
            gain = best_acc - cur_acc
            if it > 0 and gain <= 1e-6:
                break
            cur = ms[best_k][0] if cur is None else cur + ms[best_k][0]
            counts[best_k] = counts.get(best_k, 0) + 1
            cur_acc = best_acc
        say(f"=== greedy[{space}] OOF acc = {cur_acc:.4f} after {sum(counts.values())} picks ===")
        say("  weights: " + ", ".join(f"{k[0]}/{k[1]}={c}" for k, c in counts.items()))
        if best_res is None or cur_acc > best_res[1]:
            best_space, best_res = space, (counts, cur_acc, ms)
    counts, cur_acc, ms = best_res
    say(f"chosen space: {best_space} (OOF {cur_acc:.4f})")

    # ---------------- continuous weight hill-climbing refinement (real-valued weights) ----------------
    if a.refine:
        def _wacc(wd):
            lo = None
            for k, w in wd.items():
                if w <= 1e-9:
                    continue
                lo = ms[k][0] * w if lo is None else lo + ms[k][0] * w
            return float((lo.argmax(1) == y).mean()) if lo is not None else -1.0

        cur_w = {k: float(c) for k, c in counts.items()}
        cur_a = _wacc(cur_w)
        steps = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]
        improved, passes = True, 0
        while improved and passes < 10:
            improved, passes = False, passes + 1
            for k in list(cur_w):
                for s in steps:
                    for sign in (1, -1):
                        t = dict(cur_w)
                        t[k] = max(0.0, t[k] + sign * s)
                        ta = _wacc(t)
                        if ta > cur_a + 1e-6:
                            cur_w, cur_a, improved = t, ta, True
                            break
                    if improved:
                        break
            for k in ms:
                if k in cur_w or improved:
                    continue
                for w0 in (0.5, 0.25, 0.1):
                    t = dict(cur_w)
                    t[k] = w0
                    ta = _wacc(t)
                    if ta > cur_a + 1e-6:
                        cur_w, cur_a, improved = t, ta, True
                        break
        say(f"continuous refine: {cur_acc:.4f} -> {cur_a:.4f} ({len(cur_w)} members, +{cur_a - cur_acc:.4f})")
        if cur_a > cur_acc + 1e-6:
            counts = cur_w
            cur_acc = cur_a

    total = sum(counts.values())
    logits_oof = None
    test_sum = None
    for k, c in counts.items():
        logits_oof = ms[k][0] * c if logits_oof is None else logits_oof + ms[k][0] * c
        test_sum = ms[k][1] * c if test_sum is None else test_sum + ms[k][1] * c
    logits_oof = logits_oof / total
    test_logits = test_sum / total
    base_pred = logits_oof.argmax(1)
    base_acc = float((base_pred == y).mean())
    say(f"ensemble OOF acc = {base_acc:.4f}")

    # ---------------- optional triangle-specialist (Fear/Sad/Angry) blend ----------------
    if a.spec:
        try:
            TRI = (0, 1, 3)
            sd = a.spec
            sfold = np.load(os.path.join(sd, "fold_ids.npy"))
            sub_idx = np.where(np.isin(y, TRI))[0]
            if len(sfold) != len(sub_idx):
                say(f"spec blend skipped: subset size {len(sfold)} != {len(sub_idx)}")
            else:
                def _pick3(arr: np.ndarray) -> np.ndarray:
                    return arr[:, [0, 1, 3]] if arr.shape[1] == 6 else arr

                so = np.full((len(y), 3), np.nan, np.float32)
                for k in np.unique(sfold):
                    oo = os.path.join(sd, f"fold{k}_oof_{a.spec_variant}.npy")
                    if os.path.exists(oo):
                        so[sub_idx[sfold == k]] = _pick3(np.load(oo))
                st_parts = [_pick3(np.load(os.path.join(sd, f"fold{k}_test_{a.spec_variant}.npy")))
                            for k in range(5)
                            if os.path.exists(os.path.join(sd, f"fold{k}_test_{a.spec_variant}.npy"))]
                st = np.mean(st_parts, axis=0)
                st = st / np.clip(st.sum(1, keepdims=True), 1e-8, None)
                if np.isnan(so[:, 0]).all():
                    say("spec blend skipped: no specialist OOF files found")
                else:
                    def blend(p6: np.ndarray, s3: np.ndarray, w: float) -> np.ndarray:
                        q = p6.copy()
                        m = np.isin(q.argmax(1), TRI)
                        if s3.shape[0] == len(q):
                            m = m & ~np.isnan(s3[:, 0])
                        else:  # test set: s3 is full length as well; defensive
                            m = m & False
                        idx = np.where(m)[0]
                        if len(idx) == 0:
                            return q
                        tri_cols = np.array(TRI)
                        mass = q[np.ix_(idx, tri_cols)].sum(1, keepdims=True)
                        dir_main = q[np.ix_(idx, tri_cols)] / np.clip(mass, 1e-8, None)
                        spec_rows = s3[idx] if s3.shape[0] == len(q) else s3
                        new_dir = (1.0 - w) * dir_main + w * spec_rows
                        tmp = q[idx]
                        tmp[:, list(TRI)] = new_dir * mass
                        q[idx] = tmp
                        return q

                    p_oof_now = probs_from(logits_oof, best_space)
                    p_test_now = probs_from(test_logits, best_space)
                    base_acc_now = float((p_oof_now.argmax(1) == y).mean())
                    cand = {}
                    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
                        cand[w] = float((blend(p_oof_now, so, w).argmax(1) == y).mean())
                    best_w = 0.0
                    best_a = base_acc_now
                    for w in sorted(cand):
                        if cand[w] > best_a + 1e-6:
                            best_w, best_a = w, cand[w]
                    # prefer the smallest w within 0.05% of the best
                    if best_w > 0:
                        for w in sorted(cand):
                            if w < best_w and cand[w] >= best_a - 5e-4:
                                best_w = w
                                break
                    say(f"spec blend (tri[0,1,3]): w={best_w:.1f} OOF {best_a:.4f} (base {base_acc_now:.4f}) "
                        f"[candidates: " + ", ".join(f"w{w:.1f}:{acc:.4f}" for w, acc in sorted(cand.items())) + "]")
                    if best_w > 0:
                        logits_oof = to_space(blend(p_oof_now, so, best_w), best_space)
                        test_logits = to_space(blend(p_test_now, st, best_w), best_space)
                        base_pred = logits_oof.argmax(1)
                        base_acc = float((base_pred == y).mean())
        except Exception as e:
            say(f"spec blend skipped: {e}")

    # ---------------- optional logistic-regression stacker ----------------
    stack_proba_oof, stack_proba_test = None, None
    if a.stack and len(members) >= 3:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold

            keys = sorted(members)
            Xm = np.concatenate([np.log(np.clip(members[k][0], 1e-7, 1.0)) for k in keys], axis=1)
            Xt = np.concatenate([np.log(np.clip(members[k][1], 1e-7, 1.0)) for k in keys], axis=1)
            cv = StratifiedKFold(5, shuffle=True, random_state=7)
            cv_proba = np.zeros((len(y), N_CLASSES), np.float32)
            for tr, va in cv.split(Xm, y):
                lr = LogisticRegression(max_iter=3000, C=1.0)
                lr.fit(Xm[tr], y[tr])
                cv_proba[va] = lr.predict_proba(Xm[va])
            stack_acc = float((cv_proba.argmax(1) == y).mean())
            say(f"stacker (LR on log-probs, 5x CV): OOF {stack_acc:.4f}")
            if stack_acc > base_acc + 2e-3:
                lr = LogisticRegression(max_iter=3000, C=1.0)
                lr.fit(Xm, y)
                stack_proba_oof, stack_proba_test = cv_proba, lr.predict_proba(Xt).astype(np.float32)
                say("stacker preferred over the weighted average")
            else:
                say("keeping the weighted average (stacker not clearly better)")
        except Exception as e:  # pragma: no cover
            say(f"stacker skipped: {e}")

    if stack_proba_oof is not None:
        p_oof, p_test = stack_proba_oof, stack_proba_test
        base_acc = float((p_oof.argmax(1) == y).mean())
        say(f"final OOF accuracy of the chosen blend: {base_acc:.4f}")
    else:
        # ---------------- per-class bias calibration ----------------
        if a.bias:
            bias = np.zeros(N_CLASSES)
            cur = base_acc
            step_grid = [0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008]
            for _ in range(3):
                for c in range(N_CLASSES):
                    improved = True
                    while improved:
                        improved = False
                        for s in step_grid:
                            for sign in (+1, -1):
                                trial = bias.copy()
                                trial[c] += sign * s
                                acc = float(((logits_oof + trial).argmax(1) == y).mean())
                                if acc > cur + 1e-9:
                                    bias, cur, improved = trial, acc, True
                                    break
                            if improved:
                                break
            say(f"bias calibration: {np.round(bias, 3).tolist()} -> OOF {cur:.4f} ({cur - base_acc:+.4f})")
            logits_oof = logits_oof + bias
            test_logits = test_logits + bias
        p_oof = probs_from(logits_oof, best_space)
        p_test = probs_from(test_logits, best_space)
    test_pred = p_test.argmax(1).copy()
    overridden = np.zeros(len(p_test), bool)
    if a.dup:
        Xtr = np.load(os.path.join(HERE, "Xtr.npy"))[:, None].astype(np.float32)
        Xte = np.load(os.path.join(HERE, "Xte.npy"))[:, None].astype(np.float32)

        def unit(A):
            F = A.reshape(len(A), -1)
            F -= F.mean(1, keepdims=True)
            return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)

        Etr, Ete = unit(Xtr), unit(Xte)
        # OOF neighbours come from other folds only, mirroring the test-time setup
        nn_i = np.full(len(y), -1, np.int64)
        nn_s = np.zeros(len(y), np.float32)
        for k in np.unique(fold_ids):
            outside = np.where(fold_ids != k)[0]
            inside = np.where(fold_ids == k)[0]
            S = Etr[inside] @ Etr[outside].T
            j = S.argmax(1)
            nn_i[inside] = outside[j]
            nn_s[inside] = S[np.arange(len(j)), j]
        ts_i = np.empty(len(Xte), np.int64)
        ts_s = np.empty(len(Xte), np.float32)
        CH = 1024
        for s0 in range(0, len(Xte), CH):
            S = Ete[s0:s0 + CH] @ Etr.T
            j = S.argmax(1)
            ts_i[s0:s0 + CH] = j
            ts_s[s0:s0 + CH] = S[np.arange(len(j)), j]

        base_conf = p_oof.max(1)
        onehot_oof = np.zeros((len(y), N_CLASSES), np.float32)
        onehot_oof[np.arange(len(y)), y[nn_i]] = 1.0
        cands = []
        for thr in (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.90, 0.88, 0.85, 0.80):
            near = nn_s >= thr
            if not near.any():
                continue
            for w in (1.0, 0.75, 0.5, 0.25):
                blended = p_oof[near] * (1 - w) + onehot_oof[near] * w
                near_pred = blended.argmax(1)
                for conf in (1.01, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
                    m = near & (base_conf < conf)
                    pred = p_oof.argmax(1).copy()
                    pred[np.where(near)[0][m[near]]] = near_pred[m[near]]
                    acc = float((pred == y).mean())
                    cands.append((acc, int(m.sum()), thr, conf, w))
        best_acc = max(c[0] for c in cands)
        # among near-optimal rules prefer the one that touches the fewest samples
        acc, nrep, thr, conf, w = min([c for c in cands if c[0] >= best_acc - 5e-4],
                                      key=lambda c: c[1])
        say(f"dup rule: sim>={thr}, conf<{conf}, blend w={w} -> OOF {acc:.4f} "
            f"({acc - base_acc:+.4f}), replaced {nrep}/{len(y)} (best {best_acc:.4f})")
        near_te = ts_s >= thr
        blend_te = p_test.copy()
        if near_te.any():
            oh = np.zeros((len(p_test), N_CLASSES), np.float32)
            oh[np.arange(len(p_test)), y[ts_i]] = 1.0
            blend_te[near_te] = p_test[near_te] * (1 - w) + oh[near_te] * w
            blend_te /= blend_te.sum(1, keepdims=True)
        overridden = near_te & (p_test.max(1) < conf)
        test_pred[overridden] = blend_te[overridden].argmax(1)
        say(f"dup correction applied to the test set: {int(overridden.sum())} of {len(p_test)} samples")

    # pseudo-label distribution: never disagree with the final prediction
    pl_probs = p_test.copy()
    if overridden.any():
        onehot = np.zeros_like(pl_probs)
        onehot[np.arange(len(pl_probs)), test_pred] = 1.0
        pl_probs[overridden] = 0.8 * onehot[overridden] + 0.2 * p_test[overridden]
        pl_probs /= pl_probs.sum(1, keepdims=True)
    assert (pl_probs.argmax(1) == test_pred).all()

    names = open(a.names).read().split()
    assert len(names) == len(test_pred), (len(names), len(test_pred))
    with open(a.out, "w", newline="") as f:
        f.write("ID,Emotion\n")
        for n, v in zip(names, test_pred):
            f.write(f"{n},{int(v)}\n")
    dist = np.bincount(test_pred, minlength=N_CLASSES) / len(test_pred)
    say(f"wrote {a.out} ({len(test_pred)} rows)")
    say("test label distribution: " + ", ".join(f"{i}:{d:.3f}" for i, d in enumerate(dist)))
    say("train label distribution: " + ", ".join(f"{i}:{d:.3f}" for i, d in enumerate(np.bincount(y, minlength=6) / len(y))))

    if a.save_probs:
        np.save(a.save_probs, p_test.astype(np.float32))
        say(f"saved ensemble test probabilities -> {a.save_probs}")
    if a.save_pl:
        np.save(a.save_pl, pl_probs.astype(np.float32))
        say(f"saved pseudo-label probabilities -> {a.save_pl}")

    with open(a.report, "w") as f:
        f.write("\n".join(rep) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
