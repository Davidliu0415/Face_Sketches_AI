# Face Sketches AI

人脸照片与素描图像跨模态匹配项目。

本项目用于训练一个 **Face-to-Sketch Retrieval** 模型：给定一张真实人脸照片，在素描库中检索出同一身份对应的素描图像。项目采用共享的 `MobileFaceNet` 主干网络，将照片和素描映射到同一个 embedding 空间，并结合 `TripletLoss`、`BatchHardTripletLoss` 和双分支 `ArcFaceHead` 进行训练与对比实验。

## 项目目标

- 构建人脸照片到素描图像的跨模态检索系统。
- 使用共享 backbone 学习统一的人脸/素描 embedding。
- 对比 Triplet、ArcFace、Triplet + ArcFace 等训练策略。
- 输出 Top-1、Top-5、MRR、mAP、mean positive cosine 等检索指标。
- 保存 checkpoint、训练指标、测试结果和失败匹配样本，方便后续分析。

## 当前模型

```text
FaceSketchMatcher
  -> shared MobileFaceNet backbone
  -> face_head: ArcFaceHead
  -> sketch_head: ArcFaceHead
```

训练样本以 triplet 形式组织：

```text
anchor   = 人脸照片
positive = 同身份素描
negative = 不同身份素描
```

训练时主要损失由以下部分组成：

- `TripletLoss`：拉近同身份照片/素描，推远不同身份样本。
- `BatchHardTripletLoss`：在 batch 内挖掘更难的正负样本。
- `ArcFaceHead` + CE loss：分别约束人脸分支和素描分支的身份判别能力。

## 推荐阅读顺序

如果你是第一次看这个项目，建议按下面顺序阅读：

1. [docs/00_reading_order.md](docs/00_reading_order.md)：学习路线和代码阅读顺序。
2. [docs/01_project_overview.md](docs/01_project_overview.md)：项目任务、模型思路和数据流总览。
3. [docs/02_folder_and_file_guide.md](docs/02_folder_and_file_guide.md)：每个文件夹和主要代码文件的作用。
4. [docs/03_training_and_evaluation_flow.md](docs/03_training_and_evaluation_flow.md)：从 `train_val_test.py` 到训练、验证、测试的调用链。
5. [docs/04_experiment_stages.md](docs/04_experiment_stages.md)：`research_1` 到 `research_4` 的实验设计。
6. [docs/research_plan.md](docs/research_plan.md)：更完整的研究计划记录。

## 目录结构

```text
Face_Sketches_AI/
  configs/                 路径配置和训练超参数
  data/                    Dataset、manifest 处理和数据准备脚本
  network/                 MobileFaceNet、ArcFaceHead、loss 和总模型
  training/                训练、验证、检索报告和实验辅助逻辑
  eval/                    检索指标和失败案例分析
  research_1/              Triplet baseline 实验
  research_2/              ArcFace only 实验
  research_3/              ArcFace + Triplet 实验
  research_4/              CUFSF identity split 协议实验
  docs/                    项目说明、学习顺序和研究计划
  checkpoints/             训练输出目录，权重和结果文件默认写到这里
  models/                  预训练/导出模型占位目录
  logs/                    日志占位目录
  train_val_test.py        一键训练、验证、测试入口
  run_train_val_test.bat   Windows 双击运行脚本
  requirements.txt         Python 依赖
```

更细的文件说明见 [docs/02_folder_and_file_guide.md](docs/02_folder_and_file_guide.md)。

## 环境准备

建议使用 Python 3.10+ 和 PyTorch 2.x。

```bash
pip install -r requirements.txt
```

当前依赖较少：

```text
torch>=2.0
numpy>=1.24
pillow>=10.0
```

如果使用 GPU，需确保本机 PyTorch 与 CUDA 环境匹配。

## 数据准备

默认数据路径在 `configs/datasets_config.py` 中配置：

```text
data/images/structured_face_sketch_dataset
data/train_manifest.csv
data/val_manifest.csv
data/test/dataset01
data/test/dataset02
data/test/dataset03
```

训练和验证使用 CSV manifest，核心字段如下：

```csv
face_anchor,sketch_positive,sketch_negative,face_binary_label,sketch_binary_label,sketch_negative_label,identity,negative_identity
```

测试集有两类来源：

- 目录型测试集：例如 `dataset01/archive/photos` + `dataset01/archive/sketches`。
- manifest 型测试集：例如 `dataset03/processed/manifests/*.csv`。

`data/prepare_dataset03.py` 可以整理 IIIT-D Sketch Database 相关的 `dataset03` 子集：

```bash
python data/prepare_dataset03.py
```

如果只希望使用本地已有文件，不联网下载缺失来源：

```bash
python data/prepare_dataset03.py --no-download
```

数据、权重和运行结果默认不提交 Git。`.gitignore` 已忽略 `data/images/`、`data/test/`、`data/*.csv`、`checkpoints/`、`models/` 和 `logs/`。

## 快速开始

做一次不依赖真实数据的前向/反向检查：

```bash
python train_val_test.py --dry-run
```

用少量样本跑完整训练、验证、测试链路：

```bash
python train_val_test.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32
```

正式训练：

```bash
python train_val_test.py --epochs 30 --batch-size 64 --eval-batch-size 128 --amp
```

Windows 下也可以运行：

```text
run_train_val_test.bat
```

如果系统 `python` 指向 Windows Store 占位入口，`run_train_val_test.bat` 会优先寻找 `.venv`、Codex runtime、`py -3` 或系统 `python`。

## 主流程

`train_val_test.py` 是完整入口：

```text
读取参数
  -> 构建 identity label map
  -> 构建 FaceSketchMatcher
  -> 构建训练/验证 DataLoader
  -> train_one_epoch()
  -> evaluate_triplet_epoch()
  -> 保存 best.pth / last.pth
  -> retrieval_report()
  -> 写出 metrics.csv / test_results.csv / failed_matches.csv
```

输出目录默认类似：

```text
checkpoints/train_val_test_YYYYMMDD_HHMMSS/
```

常见输出文件：

```text
best.pth
last.pth
metrics.csv
test_results.csv
failed_matches.csv
retrieval_diagnostics.csv
config.json
```

## 实验阶段

项目内置了四组实验入口：

| 阶段 | 入口 | 目标 |
|---|---|---|
| `research_1` | `research_1/run_baseline.py` | Triplet baseline，只用匹配损失建立基础检索模型 |
| `research_2` | `research_2/run_arcface_only.py` | ArcFace only，观察分类约束对检索 embedding 的影响 |
| `research_3` | `research_3/run_arcface_triplet.py` | ArcFace + Triplet，联合训练当前默认方向 |
| `research_4` | `research_4/run_cufsf_protocol.py` | CUFSF 500/694 identity split 协议实验 |

详细说明见 [docs/04_experiment_stages.md](docs/04_experiment_stages.md)。

## 常用命令

只训练和验证，不跑测试：

```bash
python train_val_test.py --skip-test
```

加载已有 checkpoint，只做验证和测试：

```bash
python train_val_test.py --skip-train --checkpoint checkpoints\train_val_test_xxxxxxxx\best.pth
```

降低显存占用：

```bash
python train_val_test.py --batch-size 32 --eval-batch-size 64 --amp
```

使用梯度累积模拟更大 batch：

```bash
python train_val_test.py --batch-size 64 --accumulation-steps 2 --eval-batch-size 128 --amp
```

每 50 个 batch 打印一次训练进度：

```bash
python train_val_test.py --log-interval 50
```

## 主要参数

大部分默认参数在 `configs/params.py` 中：

```text
embedding_size = 512
batch_size = 64
eval_batch_size = 128
epochs = 30
lr = 1e-3
triplet_margin = 0.5
triplet_weight = 1.0
batch_hard_weight = 0.5
face_ce_weight = 1.0
sketch_ce_weight = 1.0
arcface_scale = 30.0
arcface_margin = 0.35
image_size = 112
```

## 评估指标

项目使用 face embedding 与 sketch embedding 的余弦相似度矩阵做检索评估：

- `Top-1`：正确素描是否排名第一。
- `Top-5`：正确素描是否出现在前五。
- `MRR`：第一个正确结果排名倒数的平均值。
- `mAP`：平均精度均值。
- `mean_positive_cosine`：正确配对的平均余弦相似度。
- `failed_matches.csv`：记录 Top-1 错误样本，方便人工分析。

## 许可证

本项目使用 [MIT License](LICENSE)。
