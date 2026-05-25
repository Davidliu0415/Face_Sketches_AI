# 人脸与素描匹配 AI 项目

本项目用于训练一个“人脸照片-素描图像”匹配模型。整体结构参考 `gc2sa_net`：配置、数据、网络、训练和评估相互独立，方便后续替换数据集、模型头或损失函数。

## 模型设计

- 主干网络：`MobileFaceNet`
- 特征维度：默认 `512`
- 特征归一化：输出 embedding 使用 L2 normalize
- 分类头：两个相互独立的 ArcFace 二分类头
  - `face_head`：人脸分支分类头
  - `sketch_head`：素描分支分类头
- 匹配损失：TripletLoss
  - anchor：人脸照片
  - positive：同身份素描
  - negative：不同身份素描

默认二分类标签为：正样本 `1`，负样本 `0`。如果后续要改成身份多分类，可以在 `configs/params.py` 中调整 `face_num_classes` 和 `sketch_num_classes`。

## 目录说明

```text
configs/
  datasets_config.py      数据路径配置
  params.py               训练超参数
data/
  dataset.py              Triplet 数据集与测试配对数据集
  train_manifest.csv      训练 triplet 清单，本地生成，不提交 Git
  val_manifest.csv        验证 triplet 清单，本地生成，不提交 Git
  images/                 训练/验证图片，本地数据，不提交 Git
  test/                   测试集 dataset01、dataset02、dataset03，本地数据，不提交 Git
network/
  mobilefacenet.py        MobileFaceNet 主干
  heads.py                ArcFace 分类头
  losses.py               TripletLoss
  face_sketch_net.py      完整匹配模型
training/
  main.py                 基础训练入口
  train.py                单轮训练与验证逻辑
eval/
  metrics.py              检索评估指标
train_val_test.py         一键训练、验证、测试脚本
run_train_val_test.bat    Windows 双击运行脚本
```

## 数据位置

当前项目默认读取以下本地路径：

```text
D:\AI\AI_V2\data\images\structured_face_sketch_dataset
D:\AI\AI_V2\data\test\dataset01
D:\AI\AI_V2\data\test\dataset02
D:\AI\AI_V2\data\test\dataset03
```

训练和验证使用 CSV manifest：

```text
D:\AI\AI_V2\data\train_manifest.csv
D:\AI\AI_V2\data\val_manifest.csv
```

manifest 的核心字段如下：

```csv
face_anchor,sketch_positive,sketch_negative,face_binary_label,sketch_binary_label,sketch_negative_label
```

相对路径会从 `configs/datasets_config.py` 中的 `image_root` 开始解析。

## 准备 dataset03

`dataset03` 使用 IIIT-D Sketch Database 目录。先运行准备脚本，它会扫描 Viewed、Semi-forensic、Forensic 子集，生成统一的 `processed/photos`、`processed/sketches` 和 `processed/manifests/*.csv`：

```bash
python data/prepare_dataset03.py
```

如果只想整理已有本地文件、不联网下载缺失源：

```bash
python data/prepare_dataset03.py --no-download
```

可选参数：

```text
--lfw-root       已下载并解压的 LFW 根目录
--lfw-archive    LFW 官方压缩包路径
--fgnet-root     已下载并解压的 FG-NET 根目录
--fgnet-archive  FG-NET 压缩包路径
--cufs-root      CUHK/CUFS 照片根目录；默认会优先复用本地 dataset02
--force-download 重新下载/覆盖已准备的外部资源
--no-download    禁止联网，只生成当前可用样本和缺失报告
```

准备完成后，测试脚本会自动发现非空 manifest，并在结果中加入 `dataset03_all` 以及可用的 dataset03 子集行。缺失、下载失败、无法校验或被跳过的样本会写入：

```text
data/test/dataset03/processed/manifests/missing_dataset03.csv
```

补图策略是官方/原始来源优先：CUHK 优先复用本地 `dataset02` 或用户提供的 CUFS 根目录；LFW 使用官方包中的 `Name/Name_0001.jpg`；FG-NET 使用用户提供的原始/可信 archive 或镜像；Forensic 只纳入 txt 中明确给出照片与素描双 URL 且能通过图像校验的样本，不自动拆分合成图。

参考来源：[IIIT-D Sketch Database](https://iab-rubric.org/old1/resources/sketchDatabase.html)、[IIIT-D README PDF](https://iab-rubric.org/images/pdf/papers/Readme_SketchDB.pdf)、[CUFS 说明](https://www.idiap.ch/software/bob/docs/bob/bob.db.cuhk_cufs/stable/index.html)、[LFW 下载规则参考](https://docs.pytorch.org/vision/0.25/_modules/torchvision/datasets/lfw.html)、[FG-NET 来源索引](https://cvhci.iar.kit.edu/429_451.php)。

## 一键训练、验证、测试

推荐直接运行：

```bash
python train_val_test.py
```

Windows 下也可以双击：

```text
run_train_val_test.bat
```

如果命令行提示 `Python was not found`，说明系统的 `python` 命令指向了 Windows Store 占位入口；此时优先使用 `run_train_val_test.bat`，它会自动寻找项目虚拟环境或 Codex 运行时里的 `python.exe`。

脚本流程：

1. 读取 `data/train_manifest.csv` 训练模型
2. 每轮使用 `data/val_manifest.csv` 验证
3. 保存 `best.pth` 和 `last.pth`
4. 使用最佳模型在 `dataset01`、`dataset02` 和已准备好的 `dataset03` 子集上做人脸到素描的 Top-1 / Top-5 检索测试

输出文件默认保存在：

```text
checkpoints/train_val_test_时间戳/
```

其中包括：

```text
best.pth
last.pth
metrics.csv
```

## 常用命令

快速检查模型能否前向与反向传播：

```bash
python train_val_test.py --dry-run
```

只跑少量样本调试完整流程：

```bash
python train_val_test.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64
```

调整 batch size 和训练轮数：

```bash
python train_val_test.py --epochs 30 --batch-size 16 --eval-batch-size 32
```

4090D 24GB 推荐使用默认配置：实际 `batch-size=64`、验证/测试 `eval-batch-size=128`，并默认开启 AMP 混合精度。这个配置比直接堆大 batch 更稳。

```bash
python train_val_test.py --batch-size 64 --eval-batch-size 128 --amp
```

如果仍然显存不足，降低实际 batch：

```bash
python train_val_test.py --batch-size 32 --eval-batch-size 64 --amp
```

如果想在显存允许的情况下模拟更大的有效 batch，可以再加梯度累积，例如实际 batch 64、累积 2 次：

```bash
python train_val_test.py --batch-size 64 --accumulation-steps 2 --eval-batch-size 128 --amp
```

每 50 个 batch 打印一次进度：

```bash
python train_val_test.py --log-interval 50
```

加载已有模型，只做验证和测试：

```bash
python train_val_test.py --skip-train --checkpoint checkpoints\train_val_test_xxxxxxxx\best.pth
```

只训练和验证，不跑测试：

```bash
python train_val_test.py --skip-test
```

## 主要参数

大部分默认值在 `configs/params.py` 中：

```text
embedding_size = 512
batch_size = 16
epochs = 30
lr = 1e-3
triplet_margin = 0.5
arcface_scale = 64.0
arcface_margin = 0.35
```

## 备注

`data/images/`、`data/test/`、`data/*.csv`、`checkpoints/`、`models/` 和 `logs/` 已在 `.gitignore` 中忽略，避免把大体积数据集和模型权重提交到 GitHub。
