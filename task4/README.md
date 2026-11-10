# 任务四：视网膜血管分割

## 任务与数据

训练集 20 张眼底图，配二值血管标签（血管为 255，背景为 0）。测试集 20 张，无标签。评价指标是 Kaggle Dice。

提交文件由官方 `segmentation_to_csv.py` 生成：把预测图缩放到 512×512，二值化后做游程编码，列名为 `Id, Predicted`。整个任务不使用外部数据和预训练权重，只用比赛给出的 20 张有标注图和 20 张无标注测试图。测试图只用在两个合法环节：测试时增强，以及用 5 折模型给测试图打伪标签后做无标签自训练，不涉及任何外部标签。

## 方法

**预处理**（`code/prep.py`）

用亮度阈值、连通域和填洞得到视野掩膜。输入 5 个通道：G、R、B 各做 CLAHE（clip=2.0，8×8 网格），加上原始 G 通道和视野掩膜。所有通道在视野内做标准化，结果缓存成 float16 的 npz。

**模型**（`code/common.py`）

4 层 U-Net，base width 32，约 1.93M 参数，卷积后接 BN，bottleneck 处加 Dropout。训练用 192×192 随机块，块中心有 50% 的概率落在血管像素上，避免细血管被背景淹没。

**训练**（`code/train.py`）

5 折交叉验证，划分固定，每折 16 张训练、4 张验证。增广包括任意角旋转、缩放、平移、翻转、弹性形变，亮度和对比度与 gamma 扰动，噪声、模糊、cutout，第二轮加上模拟病灶的亮斑暗斑。损失是 BCE（pos_weight=1.5）加 soft Dice。优化器 AdamW，学习率 3e-4 余弦退火，2500 步，批大小 8，按验证 Dice 保存 best 与 last。

**推理**（`code/predict.py`）

全卷积，测试图整图滑窗不需要切块，输出概率乘视野掩膜。8 变换 TTA（90° 旋转 4 种 × 水平翻转 2 种），5 折模型概率平均。

**级联精修器**（`code/train_refiner.py`、`code/refine_test.py`）

把 5 折的折外概率图当作第一阶段的输出，训练一个更小的第二阶段 U-Net（base 24）去纠正系统性错误，例如病灶误判、细血管漏检、边界偏移。训练输入用折外概率、标签用训练集标注，同一折的模型没有见过这些图，所以天然不泄漏。测试时对每一折的基础概率图用该折的精修器精修，再对 5 个结果求平均。

**提交编码**

预测图先在原分辨率二值化，再交给官方的 512 双线性缩放加非零即前景流程。这一步顺序不能反。如果把概率图先缩到 512 再阈值，等效于少了一次膨胀，OOF Dice 从 0.843 掉到 0.780。仓库里的 `code/segmentation_to_csv.py` 和 `code/predict.py` 中的 `rle_encode` 与官方脚本逐字节一致，已在相同输入上逐字节比对验证。

## 结果

| 项目 | 512-Dice |
|---|---|
| 单折 f0 + 8TTA，验证集 | 0.8448 |
| 5 折 OOF，best | 0.8453（阈值 0.45） |
| 5 折 OOF，last | 0.8450 |
| 级联精修器 OOF | 0.8466（阈值 0.40） |
| 精修器前单折同图 | 0.8448 |

| 提交 | 公榜 Dice |
|---|---|
| 单折 f0 + 8TTA | 0.85365 |
| 5 折模型平均 | 0.85368 |
| 精修器作用于 5 折平均概率，阈值 0.50（仓库内 `submission/submission.csv`） | 0.85729 |
| 同上，阈值 0.45 | 0.85737 |

表中的公榜 Dice 即 Kaggle 平台提交后显示的分数，`results/all_submissions_public.csv` 里记录的是同一份数据。第一张表里的 512-Dice 是本地折外或验证集分数，两者口径不同，不能直接比较。

按团队 20245627 的最终榜单记录，公榜成绩为 0.85737，排名 23/75；私榜成绩为 0.85936，排名 22/75。公榜取团队历次提交中的最好公榜成绩（阈值 0.45 的 ens-refine 版本），私榜取最终选择提交（阈值 0.50）的私榜成绩；两组数字对应不同的提交，不能直接相减比较。

阈值在 0.40 到 0.55 之间 Dice 几乎不变，0.842 到 0.845 之间波动，所以阈值不是这个任务的关键。小连通域过滤会切掉细血管，没有采用。按预测血管占比给每张图单独定阈值反而更差（OOF 0.8417）。

跨架构集成的收益很有限。v1 与精修器按概率平均后 0.8459，低于只用精修器路径的 0.8466，再加弱模型会拖累强模型，因此最终方案是单一精修器路径。

20 张图里误差最大的几张集中在视盘边缘和病变区，血管对比度低、边界模糊。不使用外部数据时，DRIVE 上公榜大致在 0.85 到 0.86 之间，继续加同构模型收益很小。

## 文件说明

```
code/prep.py             生成 5 通道预处理缓存
code/common.py           视野掩膜、预处理、U-Net、损失、整图与多尺度推理、Dice、512 重采样
code/losses.py           clDice 实现
code/train.py            单折训练
code/train_refiner.py    单折精修器训练
code/train_mtl.py        多任务辅助实验
code/eval_oof.py         5 折折外概率导出与阈值网格
code/eval_refiner.py     精修器折外评估
code/eval_variants.py    多尺度、BN 自适应等推理变体对比
code/analyze_oof.py      比较 512 提交策略
code/blend_oof.py        折外概率混合权重搜索
code/make_pseudo.py      逐折生成测试集伪标签，排除该折模型
code/make_csv.py         生成提交文件
code/make_p2.py          第二套伪标签方案
code/pp_experiments.py   后处理实验
code/avg_probs.py        概率平均
code/finetune_distill.py 蒸馏微调
code/predict.py          集成预测、PNG 输出与提交生成
code/refine_ens.py       精修器作用于平均概率
code/refine_pairs.py     成对精修对比
code/refine_test.py      测试集逐折精修并生成提交
code/submit_direct.py    直接用 API 提交
code/submit_fix.py       走签名上传地址提交，绕过无法访问的域名
code/kaggle_ipv4.py      强制 IPv4 解析的小工具
code/segmentation_to_csv.py  官方提交格式脚本
logs/                    训练日志与逐折曲线数据（log.csv）
results/all_submissions_public.csv  历次提交的公榜分数记录
data/sample_submission.csv  提交模板
submission/submission.csv   最终提交
```

## 运行方式

```bash
cd code

# 1) 预处理
python3 prep.py

# 2) 5 折基础模型，14 GB 内存的机器上同时不超过 2 折
for f in 0 1 2 3 4; do
  python3 train.py --fold $f --steps 2500 --batch 8 --patch 192 --crop 288 \
      --eval-every 250 --threads 12 --out runs_v1
done

# 3) 折外概率，用于精修器训练与阈值选择
python3 eval_oof.py --run runs_v1 --ckpt best.pt --out oof_v1.npz

# 4) 5 折精修器
for f in 0 1 2 3 4; do
  python3 train_refiner.py --fold $f --steps 2500 --batch 8 --patch 192 \
      --base 24 --ema 0.995 --blobs 0.2 --threads 12 --out runs_refiner
done

# 5) 测试集精修并生成提交
python3 refine_test.py --thr 0.50 --outdir submission_ens
```

## 注意

预处理缓存、`runs_v1` 与 `runs_refiner` 下的模型权重、折外概率 npz 都没有放入仓库，需要按上面的命令重新生成。原始数据也不在仓库内。

16 线程以上反而更慢，12 线程左右效率最高。CPU 上用 BF16 autocast 会明显拖慢训练，不要打开。

`submit_fix.py` 与 `submit_direct.py` 依赖本机 Kaggle 凭据，仓库里不含任何令牌。提交时 `kaggle competitions submit` 会卡在 `www.googleapis.com`，`submit_fix.py` 的做法是用 API 拿到签名上传地址，改走 `storage.googleapis.com` 用 curl 上传，再调用 `create_submission`。