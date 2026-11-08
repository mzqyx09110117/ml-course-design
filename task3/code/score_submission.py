"""计算提交文件的坐标 RMSE。

用法：
    python3 score_submission.py pred.csv [gt.csv]

pred.csv 的列为 id,x,y，id 形如 0081。
gt.csv 的列为 ImageID,value，ImageID 形如 0081_Fovea_X。
gt.csv 也可以通过环境变量 FOVEA_GT 指定。

gt.csv 需要自行准备，仓库里没有附带任何真值文件。
"""
import os
import sys
import numpy as np
import pandas as pd


def load_gt(path):
    gt = pd.read_csv(path)
    return dict(zip(gt['ImageID'], gt['value'].astype(float)))


def rmse(errs):
    return float(np.sqrt(np.mean(np.square(errs))))


def score(pred_csv, gt_path, verbose=True):
    gt = load_gt(gt_path)
    df = pd.read_csv(pred_csv)
    rows = []
    for _, r in df.iterrows():
        key = str(int(r['id']))
        for coord, val in (('X', float(r['x'])), ('Y', float(r['y']))):
            k = f'{key}_Fovea_{coord}'
            if k not in gt:
                raise KeyError(f'{k} 不在真值表里')
            rows.append((k, val, gt[k], val - gt[k]))
    d = pd.DataFrame(rows, columns=['key', 'pred', 'gt', 'err'])
    res = {'rmse': rmse(d['err'].values), 'n_rows': len(d)}
    if verbose:
        print(f"rows {res['n_rows']}  RMSE {res['rmse']:.5f}")
        e = d.assign(abs=d['err'].abs()).sort_values('abs', ascending=False)
        print('误差最大的几行：')
        for _, r in e.head(8).iterrows():
            print(f"  {r['key']:14s} pred {r['pred']:9.2f} gt {r['gt']:9.2f} err {r['err']:+9.2f}")
        img_err = {}
        for img in d['key'].str.split('_').str[0].unique():
            sub = d[d['key'].str.startswith(img + '_')]
            img_err[img] = float(np.hypot(*sub['err'].values)) if len(sub) == 2 else float('nan')
        print('误差最大的几张图：', ', '.join(
            f'{k}:{v:.0f}' for k, v in sorted(img_err.items(), key=lambda t: -t[1])[:6]))
    return res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    gt_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('FOVEA_GT', '')
    if not gt_path:
        raise SystemExit('需要通过第二个参数或环境变量 FOVEA_GT 指定真值表')
    score(sys.argv[1], gt_path)