#!/usr/bin/env python3
"""Push a configured training run to Kaggle as a script kernel.

Usage:
  python work/kaggle_push.py A            # push run A
  python work/kaggle_push.py A --status   # ask Kaggle for the run status
  python work/kaggle_push.py A --fetch    # download the run outputs

Runs are defined in RUNS below; each is the master script prefixed with a
`sys.argv` header so the kernel needs no command-line arguments.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASTER = os.path.join(HERE, "fer_train.py")
KAGGLE = os.path.expanduser("~/anaconda3/bin/kaggle")
USER = os.environ.get("KAGGLE_USERNAME", "")

# run key -> (slug, cli args, description)
RUNS = {
    "A": ("fer-a-scratch",
          "--arch scratch48 --size 48 --epochs 80 --bs 256 --lr 2e-3 --wd 0.02 --seed 0 "
          "--folds 5 --eval-every 20 --tta 1 --ema 1 --max-minutes 240",
          "from-scratch small resnet, 5-fold, 48px, 80 epochs"),
    "B": ("fer-b-r18",
          "--arch tv_resnet18 --size 112 --pretrained 1 --epochs 40 --bs 256 --lr 3e-4 --wd 0.01 "
          "--folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "ImageNet resnet18 @112, 5-fold"),
    "C": ("fer-c-r34",
          "--arch tv_resnet34 --size 128 --pretrained 1 --epochs 40 --bs 192 --lr 2.5e-4 --wd 0.01 "
          "--folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "ImageNet resnet34 @128, 5-fold"),
    "D": ("fer-d-scratch2",
          "--arch scratch48 --size 48 --epochs 80 --bs 256 --lr 1.5e-3 --wd 0.03 --seed 7 "
          "--folds 5 --eval-every 20 --tta 1 --ema 1 --max-minutes 240",
          "second from-scratch net (different seed/optimiser)"),
    "E": ("fer-e-effb0",
          "--arch tv_efficientnet_b0 --size 128 --pretrained 1 --epochs 40 --bs 192 --lr 3e-4 --wd 0.01 "
          "--channels-last 0 "
          "--folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "ImageNet efficientnet_b0 @128, 5-fold"),
    "P": ("fer-probe",
          "--arch scratch48 --size 48 --epochs 1 --bs 256 --lr 2e-3 --folds 5 --fold-list 0 "
          "--eval-every 1 --tta 0 --fast 1 --max-minutes 20",
          "GPU-availability probe (tiny)"),
    "F": ("fer-f-scratch-pl",
          "--arch scratch48 --size 48 --epochs 80 --bs 256 --lr 2e-3 --wd 0.02 --seed 11 "
          "--pl-file pl_probs_topk.npy --pl-thr 0.5 --pl-weight 0.5 "
          "--folds 5 --eval-every 20 --tta 1 --ema 1 --max-minutes 240",
          "from-scratch net trained with pseudo-labelled test images"),
    "G": ("fer-g-r18-pl",
          "--arch tv_resnet18 --size 112 --pretrained 1 --epochs 40 --bs 256 --lr 3e-4 --wd 0.01 "
          "--pl-file pl_probs_topk.npy --pl-thr 0.5 --pl-weight 0.5 "
          "--folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet18@112 fine-tuned with pseudo-labelled test images"),
    "H": ("fer-h-r18-96",
          "--arch tv_resnet18 --size 96 --pretrained 1 --epochs 60 --bs 256 --lr 2.5e-4 --wd 0.01 "
          "--seed 3 --folds 5 --eval-every 15 --tta 1 --ema 1 --max-minutes 300",
          "resnet18 @96, 60 epochs (scale diversity)"),
    "I": ("fer-i-r18-long",
          "--arch tv_resnet18 --size 112 --pretrained 1 --epochs 100 --bs 256 --lr 2e-4 --wd 0.01 "
          "--seed 5 --folds 5 --eval-every 25 --tta 1 --ema 1 --max-minutes 360",
          "resnet18 @112, 100-epoch schedule"),
    "J": ("fer-j-convnext",
          "--arch timm_convnext_tiny --size 112 --pretrained 1 --epochs 40 --bs 192 --lr 2e-4 "
          "--wd 0.02 --seed 0 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 360",
          "timm convnext_tiny @112 (architecture diversity)"),
    "K": ("fer-k-scratch-wide",
          "--arch scratch48 --width 96 --size 48 --epochs 80 --bs 256 --lr 1.6e-3 --wd 0.02 "
          "--seed 21 --folds 5 --eval-every 20 --tta 1 --ema 1 --max-minutes 300",
          "wider from-scratch resnet (width 96)"),
    "N": ("fer-n-r34-pl2",
          "--arch tv_resnet34 --size 128 --pretrained 1 --epochs 40 --bs 192 --lr 2.5e-4 --wd 0.01 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 2 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@128 + round-2 pseudo-labels (teacher = 6-model ensemble, OOF 0.7316)"),
    "M": ("fer-m-r34-pl",
          "--arch tv_resnet34 --size 128 --pretrained 1 --epochs 40 --bs 192 --lr 2.5e-4 --wd 0.01 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 1 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@128 fine-tuned with pseudo-labelled test images (strongest arch + PL)"),
    "L": ("fer-l-r50-128",
          "--arch tv_resnet50 --size 128 --pretrained 1 --epochs 40 --bs 160 --lr 2e-4 --wd 0.01 "
          "--seed 0 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 420",
          "ImageNet resnet50 @128 (more capacity at the best resolution)"),
    "L4": ("fer-l-r50-128-f4",
          "--arch tv_resnet50 --size 128 --pretrained 1 --epochs 40 --bs 160 --lr 2e-4 --wd 0.01 "
          "--seed 0 --folds 5 --fold-list 4 --eval-every 10 --tta 1 --ema 1 --max-minutes 120",
          "resnet50@128 fold-4 completion run (L was killed after folds 0-3)"),
    "O": ("fer-o-r34-192",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
          "--seed 0 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34 @192 (higher-resolution dimension; all prior runs were <=128)"),
    "Q": ("fer-q-r34-192-pl3",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 5 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@192 + round-3 pseudo-labels (teacher = 10-model ensemble, OOF 0.7353)"),
    "S": ("fer-s-swin-pl5",
          "--arch timm_swin_tiny_patch4_window7_224 --size 224 --pretrained 1 --epochs 40 "
          "--bs 64 --lr 2e-4 --wd 0.02 --pl-thr 0.5 --pl-weight 0.5 "
          "--seed 6 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 420",
          "swin_tiny@224 + round-5 pseudo-labels (teacher = 15-model ensemble, OOF 0.7386)"),
    "T": ("fer-t-r34-192-pl5",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 11 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@192 + round-5 pseudo-labels (teacher = 15-model ensemble, OOF 0.7386)"),
    "V": ("fer-v-swin22k-pl5",
          "--arch timm_swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --size 224 --pretrained 1 --epochs 40 "
          "--bs 64 --lr 2e-4 --wd 0.02 --pl-thr 0.5 --pl-weight 0.5 "
          "--seed 13 --folds 5 --eval-every 10 --tta 1 --ema 1 --save-ckpt 1 --max-minutes 420",
          "swin_tiny (ImageNet-22k init) @224 + round-5 pseudo-labels"),
    "Z": ("fer-z-r34-192-pl6",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 17 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@192 + round-6 pseudo-labels (teacher = 16-model ensemble 0.7395)"),
    "VY": ("fer-vy-swin22k-pl6c",
           "--arch timm_swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --size 224 --pretrained 1 --epochs 40 "
           "--bs 64 --lr 2e-4 --wd 0.02 --pl-thr 0.5 --pl-weight 0.5 "
           "--seed 29 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 420",
           "swin_tiny (ImageNet-22k) @224 + round-6 PL, seed 29 (third swin seed)"),
    "R4": ("fer-r4-r34-224-pl7",
           "--arch tv_resnet34 --size 224 --pretrained 1 --epochs 40 --bs 64 --lr 2e-4 --wd 0.01 "
           "--pl-thr 0.5 --pl-weight 0.5 "
           "--seed 41 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
           "resnet34@224 + round-7 pseudo-labels, seed 41 (new 224px scale)"),
    "P7": ("fer-p7-r34-192-pl7",
           "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
           "--pl-thr 0.5 --pl-weight 0.5 "
           "--seed 37 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
           "resnet34@192 + round-7 pseudo-labels (teacher = 20-member ensemble 0.7411), seed 37"),
    "VX": ("fer-vx-swin22k-pl6",
           "--arch timm_swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --size 224 --pretrained 1 --epochs 40 "
           "--bs 64 --lr 2e-4 --wd 0.02 --pl-thr 0.5 --pl-weight 0.5 "
           "--seed 23 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 420",
           "swin_tiny (ImageNet-22k) @224 + round-6 PL, seed 23 (second V seed)"),
    "W": ("fer-w-r34-192-pl6b",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
          "--pl-thr 0.6 --pl-weight 0.5 "
          "--seed 27 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@192 + round-6 pseudo-labels conf>=0.6, seed 27 (second Z-family seed)"),
    "Y": ("fer-y-r34-80ep-pl5",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 80 --bs 96 --lr 2e-4 --wd 0.01 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 11 --folds 5 --eval-every 20 --tta 1 --ema 1 --max-minutes 420",
          "resnet34@192 + round-5 PL, 80-epoch schedule (clean A/B vs T, seed 11)"),
    "X": ("fer-x-r34-gce-pl5",
          "--arch tv_resnet34 --size 192 --pretrained 1 --epochs 40 --bs 96 --lr 2e-4 --wd 0.01 "
          "--loss gce --gce-q 0.7 "
          "--pl-thr 0.5 --pl-weight 0.5 "
          "--seed 11 --folds 5 --eval-every 10 --tta 1 --ema 1 --max-minutes 300",
          "resnet34@192 + GCE noise-robust loss + round-5 PL (clean A/B vs T, seed 11)"),
    "U": ("fer-u-bit21k-pl5",
          "--arch timm_resnetv2_50x1_bit.goog_in21k_ft_in1k --size 192 --pretrained 1 --epochs 40 "
          "--bs 64 --lr 2e-4 --wd 0.01 --pl-thr 0.5 --pl-weight 0.5 "
          "--seed 14 --folds 5 --eval-every 10 --tta 1 --ema 1 --save-ckpt 1 --max-minutes 300",
          "BiT ResNet50x1 (ImageNet-21k init) @192 + round-5 pseudo-labels"),
}

# 折级分块变体（对抗平台长任务故障：每折一个小内核，完成即永久落袋）
# 注意：折编号是 0..4！range 必须从 0 开始（犯过 1..5 的off-by-one错）
for _k in range(0, 5):
    for _base in ("S", "T", "V", "U", "X", "Y", "Z", "W", "VX", "P7", "R4", "VY"):
        _slug, _args, _desc = RUNS[_base]
        RUNS[f"{_base}{_k}"] = (
            f"{_slug}-f{_k}",
            _args + f" --fold-list {_k}",
            f"{_desc} [chunk fold {_k}/5]",
        )

# runs that train with pseudo-labels: the probabilities are inlined into the kernel source
# (kaggle kernels push does NOT upload auxiliary files, so a base64 blob is the only
# reliable way to get them into the session).
PLB64 = {
    "F": "work/pl_probs_topk.npy",
    "G": "work/pl_probs_topk.npy",
    "M": "work/pl_probs_topk.npy",
    "N": "work/pl_probs_topk.npy",
    "Q": "work/pl_probs_topk.npy",
    "S": "work/pl_probs_topk.npy",
    "T": "work/pl_probs_topk.npy",
}
for _k in range(0, 5):
    PLB64[f"S{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"T{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"V{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"U{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"X{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"Y{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"Z{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"W{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"VX{_k}"] = "work/pl_probs_topk.npy"
    PLB64[f"P7{_k}"] = "work/pl_probs_topk7.npy"
    PLB64[f"R4{_k}"] = "work/pl_probs_topk7.npy"
    PLB64[f"VY{_k}"] = "work/pl_probs_topk.npy"


def build(run: str) -> str:
    if not USER:
        raise SystemExit("请先设置环境变量 KAGGLE_USERNAME")
    slug, args, desc = RUNS[run]
    out = f"out_{run}"
    if run in PLB64:
        import base64

        blob = base64.b64encode(open(os.path.join(ROOT, PLB64[run]), "rb").read()).decode()
        args = args.replace("--pl-file pl_probs_topk.npy", "").replace("--pl-file pl_probs.npy", "")
        args = f"{args} --pl-b64 {blob}"
        print(f"inlined pseudo-labels: {len(blob)/1024:.0f} KiB base64")
    header = (
        "import os, sys, subprocess, glob\n"
        "# test-data reliability: prefer our private re-hosted copy of the competition test\n"
        "# set (some hosts mount the competition test dir with only 4585/7065 files).\n"
        "_m = '/kaggle/input/fer-test-host'\n"
        "if os.path.isdir(os.path.join(_m, 'test')):\n"
        "    os.environ['FER_TEST_ROOT'] = os.path.join(_m, 'test')\n"
        "elif os.path.exists(os.path.join(_m, 'test.zip')):\n"
        "    subprocess.run(['unzip', '-q', os.path.join(_m, 'test.zip'), '-d', '/kaggle/working/_testdata'])\n"
        "    os.environ['FER_TEST_ROOT'] = '/kaggle/working/_testdata/test'\n"
        "elif os.path.isdir(_m):\n"
        "    _c = sorted(glob.glob(os.path.join(_m, '**', '*.jpg'), recursive=True))\n"
        "    if _c:\n"
        "        os.environ['FER_TEST_ROOT'] = os.path.dirname(_c[0])\n"
        "    else:\n"
        "        print('fer-test-host mounted but no jpgs found under', _m, '| entries:', os.listdir(_m)[:20])\n"
        "else:\n"
        "    print('fer-test-host NOT mounted; /kaggle/input =', sorted(os.listdir('/kaggle/input'))[:60])\n"
        "# fast-fail on hosts that only mount a partial competition test set: wasting ~7 min of\n"
        "# setup before the same assert costs a retry slot, so check the raw count up front.\n"
        "try:\n"
        "    _t = os.environ.get('FER_TEST_ROOT') or '/kaggle/input/competitions/2026task2/fer_data/fer_data/test'\n"
        "    _n = len(os.listdir(_t))\n"
        "    if _n < 7000:\n"
        "        print('EARLY-FAIL: test dir has', _n, 'entries:', _t)\n"
        "        sys.exit(3)\n"
        "except SystemExit:\n"
        "    raise\n"
        "except Exception as _e:\n"
        "    print('early test check skipped:', _e)\n"
        f"sys.argv = ['fer_train.py'] + '{args} --out /kaggle/working/{out}'.split()\n"
        f"# run {run}: {desc}\n"
    )
    with open(MASTER) as f:
        body = f.read()
    # a __future__ import must stay at the top: insert the header right after it
    lines = body.split("\n")
    ins = 0
    if lines and lines[0].startswith('"""'):
        for i, ln in enumerate(lines[1:], 1):
            if ln.strip().endswith('"""'):
                ins = i + 1
                break
    for i, ln in enumerate(lines):
        if ln.startswith("from __future__"):
            ins = i + 1
            break
    body = "\n".join(lines[:ins] + [header] + lines[ins:])
    kdir = os.path.join(HERE, "kaggle", run)
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "fer_train.py"), "w") as f:
        f.write(body)
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "fer_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": ["2026task2"],
        "kernel_sources": [f"{USER}/fer-test-host"],
        "model_sources": [],
    }
    import json

    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return kdir


def kaggle(*args: str) -> int:
    cmd = [KAGGLE, *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--logs", type=int, default=0)
    a = ap.parse_args()
    run = a.run.upper()
    slug = RUNS[run][0]
    ref = f"{USER}/{slug}"
    if a.status or a.logs:
        return kaggle("kernels", "status", ref)
    if a.fetch:
        dest = os.path.join(HERE, "kaggle", run, "output")
        os.makedirs(dest, exist_ok=True)
        return kaggle("kernels", "output", ref, "-p", dest)
    kdir = build(run)
    print(f"built {kdir}")
    return kaggle("kernels", "push", "-p", kdir)


if __name__ == "__main__":
    sys.exit(main())
