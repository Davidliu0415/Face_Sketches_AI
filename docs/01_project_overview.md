# 项目总览

## 任务定义

本项目研究的是 **人脸照片与素描图像匹配**。输入是一张真实人脸照片，输出是在素描库中与该照片身份相同的素描图像排序结果。

这是一个跨模态检索问题。照片和素描属于不同视觉模态：照片包含颜色、纹理和光照信息，素描更强调边缘、轮廓和局部结构。因此，模型需要学习一种统一的身份特征表示，使同一身份的照片和素描在 embedding 空间中尽量接近，不同身份尽量远离。

## 核心思路

项目使用一个共享的 `MobileFaceNet` 编码器同时处理照片和素描：

```text
face image   -> MobileFaceNet -> face embedding
sketch image -> MobileFaceNet -> sketch embedding
```

由于 backbone 共享，模型会被迫把两种模态映射到同一个特征空间。训练完成后，只需要计算人脸 embedding 与所有素描 embedding 的余弦相似度，就可以完成检索。

## 模型结构

总模型是 `network/face_sketch_net.py` 中的 `FaceSketchMatcher`：

```text
FaceSketchMatcher
  -> backbone: MobileFaceNet
  -> face_head: ArcFaceHead
  -> sketch_head: ArcFaceHead
```

`MobileFaceNet` 输出默认 512 维 embedding，并做 L2 normalize。`face_head` 和 `sketch_head` 是两个独立的 ArcFace 分类头，用于给不同模态分别施加身份分类约束。

## 训练数据形式

训练使用 triplet 样本：

```text
face_anchor      人脸照片
sketch_positive  同身份素描
sketch_negative  不同身份素描
```

`FaceSketchTripletDataset` 会从 manifest 中读取这三个路径，并返回图像张量与身份标签。

## 损失函数

训练过程中可以组合多种 loss：

- `TripletLoss`
  - anchor 是人脸照片。
  - positive 是同身份素描。
  - negative 是不同身份素描。
  - 目标是让正样本距离小于负样本距离。

- `BatchHardTripletLoss`
  - 在一个 batch 内根据身份标签寻找 hardest positive 和 hardest negative。
  - 目标是增强困难样本上的判别能力。

- `ArcFace` cross entropy
  - 人脸分支和素描分支分别做身份分类。
  - 目标是让 embedding 更具身份区分性。

最终 loss 在 `training/train.py` 中按权重相加：

```text
total_loss =
  triplet_weight * triplet_loss
  + batch_hard_weight * batch_hard_loss
  + face_ce_weight * face_ce_loss
  + sketch_ce_weight * sketch_ce_loss
```

## 测试方式

测试阶段不使用分类头做预测，而是使用 embedding 检索：

```text
所有人脸 -> face embeddings
所有素描 -> sketch embeddings
face embeddings @ sketch embeddings.T -> similarity matrix
```

每一行表示一张人脸与所有素描的相似度。按相似度排序后，就可以计算 Top-1、Top-5、MRR、mAP 等指标。

## 完整流程

```text
配置参数
  -> 读取 manifest
  -> 构建 Dataset 和 DataLoader
  -> 构建 FaceSketchMatcher
  -> 训练若干 epoch
  -> 验证 triplet loss 和检索指标
  -> 保存 best/last checkpoint
  -> 在测试集上做 face-to-sketch retrieval
  -> 写出结果 CSV 和失败样本 CSV
```

## 输出结果

默认输出目录为：

```text
checkpoints/train_val_test_YYYYMMDD_HHMMSS/
```

常见输出：

- `best.pth`：验证指标最好的 checkpoint。
- `last.pth`：最后一轮 checkpoint。
- `metrics.csv`：每轮训练和验证指标。
- `test_results.csv`：测试集检索结果。
- `failed_matches.csv`：Top-1 错误样本。
- `retrieval_diagnostics.csv`：检索诊断信息。
- `config.json`：本次运行参数、Git 状态和关键配置快照。
