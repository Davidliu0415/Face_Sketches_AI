# Face-Sketch 项目研究计划表

## 1. 研究目标

本计划表用于记录 Face-Sketch 匹配项目的三阶段研究过程：

1. 建立 Face 与 Sketch 的项目实验 baseline。
2. 测试 `face-sketch + ArcFace only`。
3. 测试 `face-sketch + ArcFace + TripletLoss`。

所有阶段统一以 Face-to-Sketch 检索效果作为最终对比目标，主要记录 `Top-1`、`Top-5`、`mean positive cosine`、训练/验证 loss 与分类/匹配准确率。

## 2. 阶段计划表

| 阶段 | 实验名称 | 核心目标 | 实验设置 | 主要记录指标 | 输出结果 | 验收标准 |
|---|---|---|---|---|---|---|
| 1 | Face-Sketch Baseline | 建立当前数据集上的可复现实验基线 | 固定 train/val/test 划分；使用基础 embedding + cosine similarity 做 Face-to-Sketch 检索；不引入 ArcFace 结论 | dataset 样本数、Top-1、Top-5、mean positive cosine、运行时间、失败样例 | baseline 结果表、baseline checkpoint 或日志、错误匹配案例 | 能在 `dataset01`、`dataset02` 上得到稳定可复现的 Top-1/Top-5 |
| 2 | Face-Sketch ArcFace Only | 验证 ArcFace 分类约束对跨模态匹配的贡献 | 使用 shared MobileFaceNet encoder；启用 face/sketch ArcFace head；关闭或置零 TripletLoss 权重 | train/val loss、face acc、sketch acc、Top-1、Top-5、mean positive cosine | ArcFace-only 实验日志、`metrics.csv`、`best.pth` | 相比 baseline 至少在 Top-1、Top-5 或 mean positive cosine 上有明确变化 |
| 3 | Face-Sketch ArcFace + TripletLoss | 验证 ArcFace 与匹配损失联合训练效果 | 使用 shared MobileFaceNet encoder + face/sketch ArcFace head + TripletLoss；Triplet 设置为 face anchor、positive sketch、negative sketch | total loss、triplet loss、triplet acc、face acc、sketch acc、Top-1、Top-5、mean positive cosine | 完整模型 checkpoint、`metrics.csv`、与前两阶段对比表 | 在检索指标上优于或可解释地接近 ArcFace-only，并完成消融对比 |

## 3. 推荐实验设置

| 阶段 | 推荐 Run ID | 推荐权重设置 | 说明 |
|---|---|---|---|
| 1 | `baseline_001` | `triplet_weight=1.0`, `face_ce_weight=0.0`, `sketch_ce_weight=0.0` | 作为无 ArcFace 分类约束的 Face-Sketch embedding baseline，用 cosine similarity 做检索评估 |
| 2 | `arcface_only_001` | `triplet_weight=0.0`, `face_ce_weight=1.0`, `sketch_ce_weight=1.0` | 只观察 ArcFace 分类约束对 embedding 检索能力的影响 |
| 3 | `arcface_triplet_001` | `triplet_weight=1.0`, `face_ce_weight=1.0`, `sketch_ce_weight=1.0` | 当前项目默认方向，验证 ArcFace 与 TripletLoss 的联合收益 |

正式对比实验应尽量保持以下配置一致：

| 字段 | 建议值 |
|---|---|
| seed | `42` |
| epochs | `30` |
| batch size | `64` |
| eval batch size | `128` |
| AMP | enabled |
| optimizer | AdamW |
| lr | `1e-3` |
| weight decay | `1e-5` |
| image size | `112` |

## 4. 实验执行记录表

| Run ID | 阶段 | 日期 | Git Commit | 配置摘要 | Train/Val/Test 数据 | Checkpoint | dataset01 Top-1 | dataset01 Top-5 | dataset02 Top-1 | dataset02 Top-5 | mean positive cosine | 观察 | 结论 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `baseline_001` | Baseline |  |  |  |  |  |  |  |  |  |  |  |  |
| `arcface_only_001` | ArcFace Only |  |  |  |  |  |  |  |  |  |  |  |  |
| `arcface_triplet_001` | ArcFace + TripletLoss |  |  |  |  |  |  |  |  |  |  |  |  |

## 5. 单次实验记录模板

复制以下模板，用于记录每一次实际运行。

```markdown
### Run ID:

- Stage:
- Date:
- Git commit:
- Command:
- Config:
  - epochs:
  - batch_size:
  - eval_batch_size:
  - lr:
  - triplet_weight:
  - face_ce_weight:
  - sketch_ce_weight:
  - arcface_scale:
  - arcface_margin:
- Dataset:
  - train samples:
  - val samples:
  - dataset01 test samples:
  - dataset02 test samples:
- Output:
  - run dir:
  - best checkpoint:
  - last checkpoint:
  - metrics.csv:
- Metrics:
  - best val loss:
  - best val triplet acc:
  - best val face acc:
  - best val sketch acc:
  - dataset01 Top-1:
  - dataset01 Top-5:
  - dataset01 mean positive cosine:
  - dataset02 Top-1:
  - dataset02 Top-5:
  - dataset02 mean positive cosine:
- Observation:
- Conclusion:
```

## 6. Sanity Check 与正式实验

每个阶段先运行小规模 sanity check，确认数据读取、前向传播、反向传播、保存 checkpoint 和测试流程都正常。

```bash
python train_val_test.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64
```

正式实验使用统一配置，确保三阶段之间可以横向比较。

```bash
python train_val_test.py --epochs 30 --batch-size 64 --eval-batch-size 128 --amp
```

如果显存不足，优先降低 batch size，并在记录表的配置摘要中注明。

```bash
python train_val_test.py --epochs 30 --batch-size 32 --eval-batch-size 64 --amp
```

## 7. 失败案例记录

| Run ID | Dataset | Face Image | Predicted Sketch | Correct Sketch | Similarity | 错误类型 | 备注 |
|---|---|---|---|---|---:|---|---|
|  |  |  |  |  |  |  |  |

建议将错误类型统一为以下几类，便于后续分析：

| 错误类型 | 说明 |
|---|---|
| identity_confusion | 检索到相似身份但不是同一人 |
| modality_gap | 照片与素描风格差异导致匹配失败 |
| pose_or_occlusion | 姿态、遮挡、表情或裁剪影响匹配 |
| low_quality_input | 输入图像质量较差 |
| annotation_or_pair_error | 数据配对或标注疑似错误 |

## 8. 最终对比表

| 阶段 | Run ID | dataset01 Top-1 | dataset01 Top-5 | dataset02 Top-1 | dataset02 Top-5 | 平均 Top-1 | 平均 Top-5 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | `baseline_001` |  |  |  |  |  |  |  |
| ArcFace Only | `arcface_only_001` |  |  |  |  |  |  |  |
| ArcFace + TripletLoss | `arcface_triplet_001` |  |  |  |  |  |  |  |

## 9. 当前假设

- Baseline 指本项目数据和评估协议下的实验基线，不是文献综述基线。
- 第三阶段的匹配损失默认使用当前项目已有的 `TripletLoss`。
- 当前项目默认配置接近第三阶段：`MobileFaceNet + dual ArcFace heads + TripletLoss`。
- 研究记录优先保存可复现实验信息，包括命令、配置、数据规模、checkpoint 路径和核心指标。
