# 文件夹与代码文件说明

这份文档按文件夹说明项目中各文件的职责。学习时可以先看每个文件夹的“核心作用”，再按需要进入具体文件。

## 顶层文件

| 文件 | 作用 |
|---|---|
| `README.md` | GitHub 首页说明，介绍项目目标、目录结构、运行方式和文档入口。 |
| `train_val_test.py` | 完整训练入口，负责训练、验证、保存 checkpoint、测试和写结果。 |
| `run_train_val_test.bat` | Windows 批处理脚本，用于自动寻找 Python 并运行 `train_val_test.py`。 |
| `requirements.txt` | Python 依赖列表。 |
| `LICENSE` | MIT 开源许可证。 |
| `.gitignore` | 忽略数据、权重、日志、缓存等不应提交的文件。 |
| `research_1.zip` | `research_1` 相关压缩包，属于辅助/备份文件。 |

## `configs/`

配置目录，负责集中管理路径和超参数。

| 文件 | 作用 |
|---|---|
| `configs/__init__.py` | Python 包标记文件。 |
| `configs/params.py` | 训练超参数，包括 batch、epoch、lr、embedding size、loss 权重、ArcFace 参数等。 |
| `configs/datasets_config.py` | 数据路径配置，包括训练/验证 manifest、测试集目录、dataset03 manifest 测试集列表。 |

重点关注：

- 想改 batch size、epoch、lr，看 `params.py`。
- 想改数据路径或测试集列表，看 `datasets_config.py`。

## `data/`

数据读取和数据准备目录。

| 文件 | 作用 |
|---|---|
| `data/__init__.py` | Python 包标记文件。 |
| `data/dataset.py` | 核心 Dataset 实现，负责读取图片、解析 manifest、构造 triplet 样本和 pair 样本。 |
| `data/create_unseen_split.py` | 根据身份重新划分 train/val manifest，用于 unseen identity 实验。 |
| `data/prepare_dataset03.py` | 整理 IIIT-D Sketch Database 相关 dataset03 子集，并生成 processed manifest。 |

`data/dataset.py` 中最重要的对象：

- `DefaultTransform`：把图片 resize 到 `image_size`，转 tensor，并归一化到 `[-1, 1]`。
- `FaceSketchTripletDataset`：训练/验证使用，返回 `face_anchor`、`sketch_positive`、`sketch_negative` 和身份标签。
- `FaceSketchPairDataset`：验证检索/测试使用，返回成对的 `face`、`sketch`、路径和 identity。
- `load_pair_dataset_from_test_set()`：根据测试集配置，从目录或 manifest 创建 pair dataset。
- `build_identity_label_map()`：从 manifest 中收集身份并生成 label id。

本地数据通常放在：

```text
data/images/
data/test/
data/train_manifest.csv
data/val_manifest.csv
```

这些路径默认被 `.gitignore` 忽略。

## `network/`

模型结构目录。

| 文件 | 作用 |
|---|---|
| `network/__init__.py` | 统一导出 `FaceSketchMatcher`、`MobileFaceNet`、`ArcFaceHead` 和 loss。 |
| `network/face_sketch_net.py` | 总模型 `FaceSketchMatcher`，组合共享 backbone 和两个 ArcFace head。 |
| `network/mobilefacenet.py` | `MobileFaceNet` 主干网络，负责提取并归一化 embedding。 |
| `network/heads.py` | `ArcFaceHead` 实现。 |
| `network/losses.py` | `TripletLoss`、`BatchHardTripletLoss` 和分类准确率函数。 |

核心结构：

```text
FaceSketchMatcher
  -> MobileFaceNet
  -> face_head
  -> sketch_head
```

## `training/`

训练、验证、检索和实验辅助逻辑。

| 文件 | 作用 |
|---|---|
| `training/__init__.py` | Python 包标记文件。 |
| `training/main.py` | 基础训练入口，同时提供 `build_model()`、`set_seed()`、`run_dry_check()`。 |
| `training/train.py` | 训练和验证主逻辑，包含 `train_one_epoch()` 和 `evaluate_triplet_epoch()`。 |
| `training/retrieval.py` | 测试/检索辅助函数，负责提取 embedding、生成检索报告和写诊断 CSV。 |
| `training/experiment.py` | 实验辅助函数，负责保存配置、读取 Git 状态、兼容加载 checkpoint。 |

最重要的文件是 `training/train.py`：

- `train_one_epoch()`：训练一轮，计算 loss，反向传播，更新参数。
- `evaluate_triplet_epoch()`：验证一轮，不更新参数，只统计 loss 和 accuracy。
- `autocast_context()`：控制 AMP 混合精度上下文。
- `TrainStats`：训练/验证指标的数据结构。

## `eval/`

评估指标目录。

| 文件 | 作用 |
|---|---|
| `eval/__init__.py` | Python 包标记文件。 |
| `eval/metrics.py` | 检索指标、失败案例和诊断统计。 |

`metrics.py` 中的关键函数：

- `cosine_scores()`：计算 face embedding 和 sketch embedding 的余弦相似度矩阵。
- `retrieval_metrics()`：计算 Top-k、MRR、mAP、mean positive cosine。
- `retrieval_failures()`：记录 Top-1 错误样本。
- `retrieval_diagnostics()`：统计重复预测、相似度分布等诊断信息。

## `research_1/`

第一阶段实验：Triplet baseline。

| 文件 | 作用 |
|---|---|
| `research_1/README.md` | baseline 实验说明和运行命令。 |
| `research_1/run_baseline.py` | 只使用 Triplet 类匹配损失，关闭 ArcFace CE loss。 |

目标是建立基础检索性能，作为后续 ArcFace 实验对照。

## `research_2/`

第二阶段实验：ArcFace only。

| 文件 | 作用 |
|---|---|
| `research_2/README.md` | ArcFace only 实验说明和运行命令。 |
| `research_2/run_arcface_only.py` | 关闭 TripletLoss，主要使用 face/sketch ArcFace CE loss。 |

目标是观察身份分类约束是否能改善跨模态检索 embedding。

## `research_3/`

第三阶段实验：ArcFace + TripletLoss。

| 文件 | 作用 |
|---|---|
| `research_3/README.md` | 联合训练实验说明和运行命令。 |
| `research_3/run_arcface_triplet.py` | 同时使用 TripletLoss、BatchHardTripletLoss 和 ArcFace CE loss。 |

这是当前项目默认方向的消融实验版本。

## `research_4/`

第四阶段实验：CUFSF identity split protocol。

| 文件 | 作用 |
|---|---|
| `research_4/run_cufsf_protocol.py` | 从 CUFSF 风格数据中收集配对样本，按身份划分训练/测试，并执行检索评估。 |

该脚本会生成本次协议专用的 train manifest 和 test pair manifest。

## `docs/`

项目文档目录。

| 文件 | 作用 |
|---|---|
| `docs/00_reading_order.md` | 推荐学习顺序。 |
| `docs/01_project_overview.md` | 项目任务、模型、训练和评估总览。 |
| `docs/02_folder_and_file_guide.md` | 本文件，说明每个文件夹和代码文件的职责。 |
| `docs/03_training_and_evaluation_flow.md` | 训练、验证、测试调用链说明。 |
| `docs/04_experiment_stages.md` | research 实验阶段说明。 |
| `docs/research_plan.md` | 原始研究计划和实验记录模板。 |

## 输出和占位目录

| 文件夹 | 作用 |
|---|---|
| `checkpoints/` | 保存训练输出、checkpoint、指标 CSV、测试结果 CSV。 |
| `logs/` | 日志占位目录。 |
| `models/` | 预训练模型或导出模型占位目录。 |
| `__pycache__/` | Python 自动生成的缓存目录，不需要手动维护。 |
| `.git/` | Git 版本管理目录，不属于项目业务代码。 |
