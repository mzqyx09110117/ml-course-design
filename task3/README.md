# 任务三：眼底黄斑中心定位

## 任务与数据

输入一张眼底图像，输出黄斑中心凹在原图中的像素坐标 `(Fovea_X, Fovea_Y)`。训练集 80 张，配 `train_location/*.xml` 形式的黄斑框标注，测试集 20 张，编号 0081–0100。图像有两种分辨率，2992×2000 的有 95 张，1956×1934 的有 5 张。

提交文件把 X 与 Y 拆成两行，共 40 行。指标是坐标 RMSE，公榜取测试集的一半评分。RMSE 对小样本非常敏感，10 张图里只要有 1 张误差 40 px，整体 RMSE 就会被抬到 13 左右，所以这个任务的关键不在平均精度，而在少数难图上不要崩。

`data/train_GT.csv` 是训练集的坐标真值，`data/sample_submission.csv` 是提交模板。

## 方法

**主模型：ResNet18 + FPN 热图回归**

- 图像缩放到 512×512，用 ImageNet 均值方差归一化。所有坐标都换算到归一化空间再算损失，因此与原始分辨率无关。
- 骨干 ResNet18（ImageNet 预训练，冻结 stem 和 layer1），FPN 解码器融合 stride 4/8/16/32 四级特征，输出 stride 4 的热图，尺寸 128×128。
- 损失由两项组成。主项是空间 softmax 交叉熵配高斯软标签（σ=1.2 个 cell），给全局位置监督；辅助项是全局 soft-argmax 得到的坐标 L1 损失，权重 0.01，直接压坐标误差。
- 推理时热图过 softmax，全局 argmax 定位峰值 cell，再在半径 6 cell 内做加权质心，得到亚像素坐标。

早期用整图 MSE 回归高斯热图，全局信号太弱，验证 RMSE 卡在 400 以上。换成空间 softmax 交叉熵后 10 轮内降到 17.5，这是整个任务里最关键的改动。

**视盘锚点通道**

用经典最亮斑检测器求视盘中心，把它的高斯热斑作为第 4 个输入通道。黄斑过曝或模糊时，模型可以借视盘位置判断大致方位。这个改动把单模型公榜 RMSE 从约 15 拉到 11.59。

**训练配置**

5 折分层交叉验证，按分辨率分层，随机种子 42。每折 60 轮，AdamW 优化器，head 学习率 1e-3，encoder 学习率 2e-4，余弦调度，批大小 8。增强含随机旋转 ±15°、缩放 0.85~1.25、平移 ±7%、水平翻转、亮度与对比度与 gamma 与通道增益抖动，以及轻度高斯模糊和噪声。

**集成**

每个模型先做多尺度推理（512 与 384 两种输入，坐标取平均），再把 7 个模型的坐标取中位数。中位数比均值稳健，个别模型在难图上跑偏时不至于把结果带偏。最终提交的 7 个模型是 5 折基线（seed 42）加两个视盘通道模型（fold0 划分，两个增强种子）。

## 结果

| 版本 | 方法 | 公榜 RMSE |
|---|---|---|
| v1 | 单模型，10 轮 | 16.60797 |
| v2 | 5 折集成，坐标中位数 | 15.51756 |
| v4 | 7 模型 + 多尺度 512/384 | 13.65279 |
| v9 | 视盘通道单模型 fold0 | 11.59175 |
| v10 | 5 折 + disc_f0 | 12.69115 |
| v11 | 5 折 + disc_f0 + disc_f0_s2（最终） | 11.99623 |

本地 5 折单模型交叉验证 RMSE 19.7 px，误差中位数 7.9 px，最好一折 6.1 px，最大误差 134 px。也就是说大部分图能落到 5 px 以内，整体 RMSE 完全由个别图决定。

最终提交 `submission/submission.csv` 就是 v11，对应 40 行共 20 张图的坐标，公榜 11.99623。

按团队 20245627 的最终榜单记录，公榜成绩为 11.59175，排名 28/78；私榜成绩为 19.90697，排名 29/78。公榜取团队历次提交中的最好公榜成绩（v9），私榜取最终选择提交（v11）的私榜成绩；两组数字对应不同的提交，不能直接相减比较。

## 试过但没有采用的方案

| 方案 | 结果 |
|---|---|
| 整图 MSE 回归高斯热图 | 全局信号弱，交叉验证 RMSE 停在 400 以上 |
| 直接坐标回归（ResNet18 + 全局池化） | 第 10 轮验证误差 134 px，收敛太慢 |
| 翻转、旋转、颜色 TTA | 困难图预测不稳定，明显变差 |
| 二阶高分辨率裁剪精修 | 验证 28 px，组合后反而变差 |
| CLAHE 增强 | 交叉验证持平，公榜 13.68 |
| 768 或 640 输入分辨率 | 768 验证 20.0，变差 |
| ResNet34 骨干 | fold0 验证 15.0，与 ResNet18 的 15.3 持平 |
| 模板匹配（归一化相关） | 交叉验证中位数 17.7 px，不如卷积网络 |
| 暗区质心回退、置信度收缩 | 交叉验证无改善 |
| 第二组 5 折（seed 123） | 单模型波动大，集成后公榜 13.73 |

多尺度（512 + 384）坐标平均是少数有效的技巧，交叉验证从 19.67 降到 18.87，因此保留。

## 文件说明

```
code/fovea_lib.py         模型、数据集、增强、热图工具、视盘通道
code/train_fovea.py       单折或全量训练（--fold -1 为全量，--disc_ch 开启视盘通道）
code/predict.py           推理与提交生成
code/cv_collect.py        收集各折验证集预测
code/cv_eval.py           5 折交叉验证评估
code/hm_fuse.py           热图级融合实验
code/hm_agg.py            热图聚合实验
code/prior_fusion_cv.py   先验融合的交叉验证
code/eval_stage12.py      二阶方案评估
code/score_submission.py  用给定的坐标 CSV 复算提交的 RMSE
code/submit_kaggle.py     提交到 Kaggle
code/train_y.py           纵向坐标专项实验
code/train_coord.py       直接坐标回归实验
code/blend_test.py        坐标级混合实验
code/stage2_lib.py        二阶精修实验库
code/train_stage2.py      二阶精修训练
code/tools/               对照实验脚本，包括多尺度、TTA、视盘分析、模板匹配、置信度分析
logs/                     训练与对照实验日志
data/train_GT.csv         训练集坐标真值
data/sample_submission.csv 提交模板
submission/submission.csv  最终提交（v11）
训练过程详解.md           完整的方案演进与实验记录
```

## 运行方式

```bash
# 1) 训练 5 折基线
for f in 0 1 2 3 4; do
  python3 code/train_fovea.py --name ce_f${f}_60 --fold $f --epochs 60
done

# 2) 训练视盘通道模型，fold0 划分，两个增强种子
python3 code/train_fovea.py --name disc_f0    --fold 0 --epochs 60 --disc_ch 1
python3 code/train_fovea.py --name disc_f0_s2 --fold 0 --epochs 60 --disc_ch 1 --aug_seed 2

# 3) 生成最终提交
python3 code/predict.py \
  --ckpts runs/ce_f0_60/best.pt runs/ce_f1_60/best.pt runs/ce_f2_60/best.pt \
          runs/ce_f3_60/best.pt runs/ce_f4_60/best.pt \
          runs/disc_f0/best.pt runs/disc_f0_s2/best.pt \
  --no_flip --scales 1.0,0.75 --agg median --out submission.csv

# 4) 如有本地坐标真值表，可复算提交的 RMSE
python3 code/score_submission.py submission.csv /path/to/gt.csv
```

## 注意

数据集、图像、模型权重都没有放进仓库。训练脚本要求把眼底图像按编号放到约定目录，checkpoint 需要自己重训生成。历次提交的完整归档同样没有放入。

`code/score_submission.py` 只是把提交文件和一份外部坐标真值表对齐后算 RMSE，不参与任何训练、融合或提交生成。