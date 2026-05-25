# 实验阶段说明

本项目除了主入口 `train_val_test.py`，还提供了 `research_1` 到 `research_4` 四组实验脚本。它们大多复用同一套数据、模型、训练和评估模块，主要区别在实验目标、loss 权重和评估协议。

## 总体关系

```text
shared modules
  -> configs/
  -> data/
  -> network/
  -> training/
  -> eval/

experiment scripts
  -> research_1/run_baseline.py
  -> research_2/run_arcface_only.py
  -> research_3/run_arcface_triplet.py
  -> research_4/run_cufsf_protocol.py
```

## Stage 1: Triplet Baseline

目录：

```text
research_1/
```

入口：

```bash
python research_1/run_baseline.py
```

目标：

- 建立最基础的 Face-to-Sketch embedding 检索 baseline。
- 主要依赖 TripletLoss。
- 关闭或弱化 ArcFace 分类损失。

典型权重：

```text
triplet_weight = 1.0
face_ce_weight = 0.0
sketch_ce_weight = 0.0
```

输出目录：

```text
checkpoints/research_1/
```

## Stage 2: ArcFace Only

目录：

```text
research_2/
```

入口：

```bash
python research_2/run_arcface_only.py
```

目标：

- 测试单独使用 ArcFace 分类约束是否能学习到有效的跨模态 embedding。
- TripletLoss 不作为训练目标，只保留为观察指标。

典型权重：

```text
triplet_weight = 0.0
face_ce_weight = 1.0
sketch_ce_weight = 1.0
```

输出目录：

```text
checkpoints/research_2/
```

## Stage 3: ArcFace + TripletLoss

目录：

```text
research_3/
```

入口：

```bash
python research_3/run_arcface_triplet.py
```

目标：

- 联合使用匹配损失和身份分类损失。
- 对比 Stage 1 和 Stage 2，观察联合训练是否提升检索效果。

典型权重：

```text
triplet_weight = 1.0
face_ce_weight = 1.0
sketch_ce_weight = 1.0
batch_hard_weight = params.batch_hard_weight
```

输出目录：

```text
checkpoints/research_3/
```

## Stage 4: CUFSF Identity Split Protocol

目录：

```text
research_4/
```

入口：

```bash
python research_4/run_cufsf_protocol.py
```

目标：

- 使用更严格的身份划分协议。
- 默认从测试数据目录收集 face/sketch pair。
- 将身份划分为训练身份和测试身份，例如 500 个身份训练，其余身份测试。
- 在 unseen identity 上评估检索能力。

典型流程：

```text
collect_pairs()
  -> split_pairs()
  -> build_triplet_rows()
  -> write train manifest
  -> train_one_epoch()
  -> run_cufsf_test()
```

输出目录：

```text
checkpoints/research_4/
```

常见输出：

- `cufsf_train_manifest.csv`
- `cufsf_test_pairs.csv`
- `split_summary.json`
- `metrics.csv`
- `cufsf_results.csv`
- `failed_matches.csv`

## 推荐实验顺序

建议按下面顺序跑实验：

1. 先跑主入口 dry run。

```bash
python train_val_test.py --dry-run
```

2. 再用少量样本跑主流程。

```bash
python train_val_test.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32
```

3. 跑 Stage 1 baseline。

```bash
python research_1/run_baseline.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32 --no-amp
```

4. 跑 Stage 2 ArcFace only。

```bash
python research_2/run_arcface_only.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32 --no-amp
```

5. 跑 Stage 3 ArcFace + Triplet。

```bash
python research_3/run_arcface_triplet.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32 --no-amp
```

6. 如果 dataset01 / CUFSF 风格数据完整，再跑 Stage 4。

```bash
python research_4/run_cufsf_protocol.py --epochs 1 --limit-train 64 --limit-test 64 --no-amp
```

## 结果对比建议

建议最终比较以下指标：

| Stage | Top-1 | Top-5 | MRR | mAP | mean positive cosine | 备注 |
|---|---:|---:|---:|---:|---:|---|
| Stage 1 Triplet |  |  |  |  |  |  |
| Stage 2 ArcFace |  |  |  |  |  |  |
| Stage 3 Joint |  |  |  |  |  |  |
| Stage 4 CUFSF |  |  |  |  |  |  |

同时建议查看 `failed_matches.csv`，因为 Top-k 指标只能告诉你错了多少，失败样本能帮助判断错误类型，例如身份相似、模态差异、姿态遮挡、低质量输入或配对标注问题。
