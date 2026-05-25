# 训练与评估调用链

这份文档说明 `train_val_test.py` 如何串起配置、数据、模型、训练、验证和测试。

## 一键入口

主入口是：

```bash
python train_val_test.py
```

该脚本负责：

1. 解析命令行参数。
2. 设置随机种子。
3. 构建身份标签映射。
4. 创建模型、loss、optimizer、DataLoader。
5. 执行训练和验证。
6. 保存 checkpoint 和指标。
7. 在可用测试集上做检索评估。

## 顶层调用链

```text
train_val_test.py
  -> parse_args()
  -> set_seed()
  -> build_identity_label_map()
  -> build_model()
  -> FaceSketchTripletDataset()
  -> DataLoader
  -> train_one_epoch()
  -> evaluate_triplet_epoch()
  -> retrieval_report()
  -> save_checkpoint()
```

## 模型构建

模型由 `training/main.py` 中的 `build_model()` 创建：

```text
build_model()
  -> FaceSketchMatcher
       -> MobileFaceNet
       -> ArcFaceHead(face)
       -> ArcFaceHead(sketch)
```

`FaceSketchMatcher.embed(images)` 是训练和测试阶段最常用的方法，它会调用共享的 `MobileFaceNet` 并返回归一化 embedding。

## 训练 DataLoader

训练数据来自：

```text
FaceSketchTripletDataset
  -> torch.utils.data.DataLoader
```

每个 batch 主要包含：

```text
face_anchor
sketch_positive
sketch_negative
face_identity_label
sketch_identity_label
sketch_negative_identity_label
face_binary_label
sketch_binary_label
sketch_negative_label
```

其中 identity label 用于 ArcFace 和 BatchHardTripletLoss，binary label 保留了较早实验中的正负样本标签。

## 单轮训练

核心函数是 `training/train.py` 中的 `train_one_epoch()`。

每个 batch 的主要流程：

```text
读取 face_anchor / sketch_positive / sketch_negative
  -> model.embed(face_anchor)
  -> model.embed(sketch_positive)
  -> model.embed(sketch_negative)
  -> TripletLoss(anchor, positive, negative)
  -> BatchHardTripletLoss(all_embeddings, all_identity_labels)
  -> face_head(face_embedding, face_labels)
  -> sketch_head(sketch_embeddings, sketch_labels)
  -> total loss
  -> backward
  -> optimizer.step
```

训练统计会保存到 `TrainStats`：

```text
loss
triplet_loss
batch_hard_loss
face_ce_loss
sketch_ce_loss
face_acc
sketch_acc
triplet_acc
grad_norm
```

## 验证流程

验证函数是 `evaluate_triplet_epoch()`。

它与训练流程基本一致，但有三个区别：

- 使用 `model.eval()`。
- 使用 `torch.no_grad()`。
- 不执行 backward 和 optimizer step。

如果验证集身份不在训练身份表中，代码会关闭验证阶段的 ArcFace CE loss，只保留 triplet 类指标，避免分类头类别数不匹配。

## 检索验证与测试

检索流程由 `training/retrieval.py` 和 `eval/metrics.py` 共同完成：

```text
FaceSketchPairDataset
  -> DataLoader
  -> embed_pair_dataset()
       -> model.embed(face)
       -> model.embed(sketch)
  -> retrieval_metrics()
       -> cosine similarity matrix
       -> Top-k / MRR / mAP
  -> retrieval_failures()
  -> retrieval_diagnostics()
```

测试集配置在 `configs/datasets_config.py` 的 `TEST_SETS` 中。项目支持两种测试集：

- `directories`：照片目录 + 素描目录。
- `manifest`：包含 `face_anchor`、`sketch_positive`、`identity` 的 CSV。

`data_config.get_test_sets()` 会自动过滤不可用测试集。也就是说，只有目录存在或 manifest 非空的测试集才会参与测试。

## Checkpoint 逻辑

训练时会保存：

```text
best.pth
last.pth
```

`best.pth` 的选择逻辑：

- 如果启用了 validation retrieval，则优先根据验证集 `mrr` 选择最佳模型。
- 如果没有 validation retrieval，则根据 `val_loss` 选择最佳模型。

训练结束后，如果存在 `best.pth`，测试阶段会加载最佳模型；否则使用最后一轮模型。

## 输出文件

每次运行会创建一个时间戳目录：

```text
checkpoints/train_val_test_YYYYMMDD_HHMMSS/
```

常见输出：

| 文件 | 内容 |
|---|---|
| `config.json` | 命令行参数、Git 状态、训练配置快照。 |
| `metrics.csv` | 每轮训练/验证指标。 |
| `best.pth` | 最佳 checkpoint。 |
| `last.pth` | 最后一轮 checkpoint。 |
| `test_results.csv` | 每个测试集的 Top-k、MRR、mAP 等结果。 |
| `failed_matches.csv` | Top-1 错误样本。 |
| `retrieval_diagnostics.csv` | 重复预测等诊断信息。 |

## Dry Run

快速检查模型能否前向和反向传播：

```bash
python train_val_test.py --dry-run
```

该模式使用随机张量，不依赖真实数据集。
