# 任务二：图像情感分类

## 任务与数据

赛题给一张 48×48 灰度人脸图，判断它属于 6 类情感之一。类别按官方编号 0~5 依次是愤怒、恐惧、快乐、悲伤、惊讶、中性。训练集 28,275 张，测试集 7,065 张，评分用测试集准确率。

训练集按类别文件夹组织，文件夹名按字母序排列，和官方数字编号并不一致，代码里用 `CLASS_ORDER` 统一成官方顺序，避免提交时标签错位，这一步弄反的话本地 OOF 完全正常，提交分数会掉一大截。

数据本身带噪声。28,275 张图只有 27,081 个不同的像素指纹，有 1,000 个重复组、涉及 2,194 张图，其中 39 组内部标签不一致。测试集里有 520 张图与训练集完全相同，按像素余弦相似度不低于 0.99 的测试图占 7.8%。训练集和测试集同分布，本地 5 折 OOF 可以用来选模型。

## 方法

**数据检查**（`code/00_eda_dupes.py`、`code/01_near_dup.py`）

统计类别分布、逐像素重复组、测试集与训练集的近重复比例，确认没有分布漂移。

**训练**（`code/fer_train.py`）

所有模型共用一套固定的 5 折划分，每折训练时验证折不参与，5 折跑完每个训练样本都有一个没见过它的模型给出的预测，即 OOF。骨干有四类。

- 从零训练的小 ResNet，stem width 64 或 96，输入 48×48。
- ImageNet 预训练 ResNet18/34/50，灰度复制成 3 通道，输入 112、128、192。
- Swin-Tiny，1k 与 22k 两版预训练权重，输入 224。
- BiT-R50x1（21k 预训练），输入 192。

优化器 AdamW，余弦学习率带 3% warmup，权重衰减 0.01~0.03，批大小 96，梯度裁剪 5.0，混合精度加 channels_last。预训练模型训 40 轮，从零模型训 80 轮。EMA 衰减 0.999，从第 10 轮重新初始化。每个运行同时保存 raw 与 EMA 两套权重在验证集和测试集上的概率，两套都作为独立成员参加融合。增强全部在 GPU 上批量完成，含随机仿射、水平翻转、亮度与对比度抖动、随机遮挡，另外用 Mixup（α=0.2）和标签平滑 0.1。

**伪标签**（`code/pl_select.py`）

用当前集成的测试集概率打标。按训练集类别先验给每类分配名额，类内取置信度最高的样本，置信度下限 0.6，默认选 3,500 张，约占测试集的 49.5%，伪标签样本在损失里权重 0.5。每轮伪标签后完整重跑 5 折得到新的 OOF，只有优于同架构监督模型的轮次才进集成。

**推理**：10 组 TTA，原图、水平翻转，以及两者向上、下、左、右各平移 2 像素的视图，概率取平均。

**融合**（`code/ensemble.py`）

22 个训练运行，每运行 raw 与 EMA 两套概率，共 44 个成员。先做带放回的贪心选择，概率空间 OOF 0.7396，对数概率空间 0.7408，选后者。再以整数权重为起点，按 0.8 到 0.025 的步长做爬山，OOF 到 0.7415，选中 10 个成员。之后给 6 个类别各配一个对数偏置做坐标上升，偏置为 [0, 0, -0.06, 0.008, 0.045, 0]，OOF 到 0.7417。最后是近重复校正，规则是像素余弦相似度不低于 0.99 且模型置信度低于 0.5 时用最近训练图的标签替换，这条规则在 OOF 上只改了 24 个样本。

另外用 44 个成员的对数概率拼成 264 维特征训了 Logistic 回归堆叠，OOF 0.7351，低于加权平均，没有采用。

## 结果

| 项目 | 数值 |
|---|---|
| 从零训练小 ResNet（run A） | 0.6968 |
| ResNet18 预训练（run B） | 0.6970 |
| A + B 融合 | 0.7177 |
| 单模型最强（ResNet34@192 + 5 轮伪标签，EMA） | 0.7191 |
| 贪心集成，对数概率空间（7 个成员） | 0.7408 |
| 加连续权重优化 | 0.7415 |
| 加类别偏置校准（最终 v9） | 0.7417 |
| 再加近重复校正 | 0.7416，最高一组设置 0.7420 |

OOF 混淆矩阵对角线即各类召回率：愤怒 67.6%，恐惧 55.7%，快乐 91.4%，悲伤 63.1%，惊讶 81.6%，中性 75.8%。主要混淆集中在恐惧、悲伤、愤怒之间，例如悲伤被判为中性 800 张，恐惧被判为悲伤 637 张。48×48 分辨率下这三类的面部肌肉变化很接近，加上重复图里有 39 组标签冲突，剩余错误里有相当一部分由数据噪声决定。

最终提交的预测分布为愤怒 13.3%、恐惧 11.4%、快乐 26.2%、悲伤 17.0%、惊讶 11.5%、中性 20.6%，与训练集类别先验一致。

平台公榜与私榜成绩均为 0.749，公榜排名 11/74，私榜排名 11/74。

## 文件说明

```
code/00_prep.py         从原始类别文件夹构建 numpy 数组
code/00_eda_dupes.py    类别分布与完全重复检查
code/01_near_dup.py     测试集与训练集的近重复分析
code/fer_train.py       单模型单折训练主脚本（含 EMA、TTA、伪标签接口）
code/kaggle_push.py     按运行代号把训练任务推成 Kaggle 内核，并可取回产物
code/merge_chunks.py    合并分块输出
code/pl_select.py       生成按类别先验配额的伪标签文件
code/ensemble.py        多成员融合、权重搜索、偏置校准、生成提交
code/summary.py         汇总各运行的配置与 OOF
code/verify_sub.py      提交文件格式校验
code/submit_patched.py  提交辅助脚本，处理当时环境下的域名解析问题
data/ytr.npy            训练集标签数组（28,275 个 0~5 的整数）
data/test_names.txt     测试集文件名顺序
reports/ens_v7_final.txt    集成搜索过程记录
reports/ens_v8_full.txt     集成搜索过程记录
reports/ens_v9_p7.txt       v9 集成搜索记录，含 44 个成员的单体 OOF
reports/ens_v9_try.txt      调参过程记录
reports/ens_v9_vx.txt       最终 v9 的完整记录，含偏置与近重复校正
submission/submission.csv   最终提交，取自 task2/work/sub_v9_vx.csv
```

## 运行方式

```bash
# 数据准备，原始数据放在 work/fer_data 下
python3 code/00_prep.py
python3 code/00_eda_dupes.py

# 训练一个运行的某一折，其余参数决定骨干与输入尺寸
python3 code/fer_train.py --arch tv_resnet34 --size 192 --pretrained 1 \
    --epochs 40 --bs 96 --lr 2.5e-4 --wd 0.01 --folds 5 --fold 0 --ema 1 --out runs/C

# 推到 Kaggle 上跑并取回产物（需要本机配置好 Kaggle 凭据）
python3 code/kaggle_push.py C
python3 code/kaggle_push.py C --fetch

# 生成伪标签
python3 code/pl_select.py --probs runs/C/oof.npy --y data/ytr.npy \
    --budget 3500 --floor 0.6 --out runs/pl_3500.txt

# 融合与提交
python3 code/ensemble.py --runs runs/C runs/T runs/Q runs/V \
    --y data/ytr.npy --names data/test_names.txt --out submission.csv
python3 code/verify_sub.py
```

最终提交 `submission/submission.csv` 来自融合脚本对 v9_vx 配置的输出（原始文件 `task2/work/sub_v9_vx.csv`），共 7,065 行，预测类别分布为愤怒 13.3%、恐惧 11.4%、快乐 26.2%、悲伤 17.0%、惊讶 11.5%、中性 20.6%。

## 注意

`kaggle_push.py` 和 `submit_patched.py` 依赖本机的 Kaggle 凭据（`~/.kaggle/kaggle.json` 或 `KAGGLE_USERNAME`、`KAGGLE_KEY` 环境变量），仓库中不含任何令牌。`kaggle_push.py` 里的账号名从环境变量读取，运行前需要先设置 `KAGGLE_USERNAME`。`submit_patched.py` 里的域名映射是为当时网络环境临时加的，换环境后不需要保留。各运行的训练产物（概率数组、模型权重）体积大，没有放进仓库，只保留了集成搜索的文字记录。