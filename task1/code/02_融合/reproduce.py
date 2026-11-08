# -*- coding: utf-8 -*-
"""植物分类竞赛（任务一）—— 融合与推理，生成提交文件。

方法概要
--------
15 个预训练模型分别做 5 折交叉验证，对训练中没有见过的折产生预测（OOF）。
在 OOF 上使用贪心前向选择挑选融合权重，再用同一套权重对测试集概率加权平均，
取 argmax 得到预测。

结果
----
  OOF micro-F1    0.9600

用法:
    cd task1
    python3 code/02_融合/reproduce.py
    python3 code/02_融合/reproduce.py --verify
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
N_STEPS = 25
TOL = 1e-6


def load_labels():
    """读取训练集标签。"""
    df = pd.read_csv(ROOT / "inputs" / "train_labels.csv")
    return df


def load_models():
    oof_dir, test_dir = ROOT / "inputs" / "oof", ROOT / "inputs" / "test"
    models = {}
    for f in sorted(oof_dir.glob("*_oof.npy")):
        key = f.name[:-8]                       # 去掉 _oof.npy
        tf = test_dir / f"{key}_test.npy"
        if not tf.exists():
            print(f"  ! {key} 缺少 test.npy，跳过")
            continue
        models[key] = (np.load(f), np.load(tf))
    return models


def greedy_select(P, y, steps=N_STEPS):
    """贪心前向选择（带重复）：每步挑一个最能提升 OOF micro-F1 的模型加入。"""
    cur = np.zeros_like(P[0])
    sel = []
    for _ in range(steps):
        best_i, best_v = None, -1.0
        for i in range(len(P)):
            cand = (cur * len(sel) + P[i]) / (len(sel) + 1)
            v = f1_score(y, cand.argmax(1), average="micro")
            if v > best_v:
                best_i, best_v = i, v
        cur = (cur * len(sel) + P[best_i]) / (len(sel) + 1)
        sel.append(best_i)
    return cur, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = ap.parse_args()

    df = load_labels()
    y = df["label"].to_numpy()
    classes = list(pd.unique(df.sort_values("label")["class"]))
    n_train = len(y)
    print(f"训练标签 {n_train} 条，{len(classes)} 类: {classes}")

    models = load_models()
    print(f"可用基模型 {len(models)} 个（每个含 OOF 与测试概率）")
    keys = list(models)

    P = np.stack([models[k][0] for k in keys])
    T = np.stack([models[k][1] for k in keys])
    assert P.shape[1] == n_train, f"OOF 行数 {P.shape[1]} != 训练样本数 {n_train}"

    print("\n各基模型的 OOF 分数（模型没见过的折上评估）:")
    for i, k in enumerate(keys):
        print(f"  {k:44s} {f1_score(y, P[i].argmax(1), average='micro'):.4f}")

    # ---- 贪心选择，只在 OOF 上 ----
    ens_oof, sel = greedy_select(P, y)
    oof_f1 = f1_score(y, ens_oof.argmax(1), average="micro")
    print(f"\n贪心融合 OOF micro-F1 = {oof_f1:.4f}")

    from collections import Counter
    weights = Counter(keys[i] for i in sel)
    print("选出的权重（25 步内的出现次数）:")
    for k, v in weights.most_common():
        print(f"  {k:44s} x{v}")

    ens_test = np.mean([T[i] for i in sel], axis=0)

    # ---- 自检：OOF 上的混淆矩阵 ----
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y, ens_oof.argmax(1)):
        cm[t, p] += 1
    print("\nOOF 混淆矩阵（行=真实，列=预测）:")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    if len(classes) == 5:
        print(f"Black-grass <-> Loose Silky-bent 互误: {cm[0,2] + cm[2,0]} 次")

    # ---- 输出 ----
    ids = (ROOT / "inputs" / "test_ids.txt").read_text(encoding="utf-8").split()
    assert ens_test.shape[0] == len(ids), f"测试概率行数 {ens_test.shape[0]} != {len(ids)}"
    pred = ens_test.argmax(1)

    tpl = pd.read_csv(ROOT / "submission_template_official.csv")
    mapped = dict(zip(ids, [classes[i] for i in pred]))
    out = pd.DataFrame({"ID": tpl["ID"].tolist(),
                        "Category": [mapped[i] for i in tpl["ID"].tolist()]})

    checks = [
        ("行数 == 378", len(out) == 378),
        ("列名为 ID,Category", list(out.columns) == ["ID", "Category"]),
        ("无空值", bool(out.notna().all().all())),
        ("ID 顺序与官方模板一致", out["ID"].tolist() == tpl["ID"].tolist()),
        ("类别全部合法", set(out["Category"]) <= set(classes)),
        (f"预测不止一类（{out['Category'].nunique()} 类）", out["Category"].nunique() > 1),
    ]
    print("\n提交文件自检:")
    ok = True
    for n, v in checks:
        print(f"  {'OK  ' if v else 'FAIL'} {n}")
        ok &= v
    if not ok:
        raise SystemExit("自检未通过")

    out.to_csv(args.out, index=False)

    # ---- 保存选定配置 ----
    cfg = {
        "oof_micro_f1": float(oof_f1),
        "model_weights": {k: int(v) for k, v in weights.items()},
        "n_steps": N_STEPS,
        "classes": classes,
    }
    (ROOT / "config" / "selected_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已写出 {args.out}")
    print(f"已写出 config/selected_config.json")
    print(f"\n预测分布:\n{out.Category.value_counts().to_string()}")

    if args.verify:
        ref = ROOT / "submission.csv"
        if Path(args.out).resolve() == ref.resolve():
            print("\n（--verify 需配合 --out 输出到别处）")
        else:
            a = pd.read_csv(ref)
            same = (a["ID"].tolist() == out["ID"].tolist() and
                    a["Category"].tolist() == out["Category"].tolist())
            print(f"\n与 submission.csv 比对: {'完全一致' if same else '不一致'}")
            if not same:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
