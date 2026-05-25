# 推荐阅读顺序

这份文档用于帮助你从整体到细节学习本项目。建议不要一开始就直接读训练循环，先把数据、模型和入口脚本之间的关系建立起来。

## 第一遍：建立地图

1. `README.md`
   - 先看项目目标、目录结构、快速运行命令。
   - 目标是知道这个项目要解决什么问题，以及主入口在哪里。

2. `docs/01_project_overview.md`
   - 看任务定义、模型设计、训练目标和评估方式。
   - 目标是理解 Face-to-Sketch Retrieval 的整体思路。

3. `docs/02_folder_and_file_guide.md`
   - 看每个文件夹和主要代码文件的职责。
   - 目标是知道以后要查某个功能应该去哪一个文件。

## 第二遍：按代码执行顺序读

1. `configs/datasets_config.py`
   - 先看数据路径和测试集配置。

2. `configs/params.py`
   - 再看模型和训练超参数。

3. `data/dataset.py`
   - 理解 `FaceSketchTripletDataset` 和 `FaceSketchPairDataset` 返回的数据结构。

4. `network/face_sketch_net.py`
   - 看总模型 `FaceSketchMatcher` 如何组合 backbone 和两个 ArcFace head。

5. `network/mobilefacenet.py`
   - 看共享特征提取器如何输出 embedding。

6. `network/heads.py` 和 `network/losses.py`
   - 看 ArcFace、TripletLoss、BatchHardTripletLoss 的实现。

7. `training/train.py`
   - 看 `train_one_epoch()` 和 `evaluate_triplet_epoch()`。

8. `training/retrieval.py` 和 `eval/metrics.py`
   - 看测试阶段如何编码图片、计算相似度和检索指标。

9. `train_val_test.py`
   - 最后回到完整入口，把训练、验证、测试串起来。

## 第三遍：看实验脚本

1. `research_1/run_baseline.py`
   - Triplet baseline。

2. `research_2/run_arcface_only.py`
   - ArcFace only。

3. `research_3/run_arcface_triplet.py`
   - ArcFace + Triplet。

4. `research_4/run_cufsf_protocol.py`
   - CUFSF identity split 协议。

这些脚本复用了主项目的数据、模型、训练和评估模块，主要区别是实验协议、loss 权重和输出格式。

## 最短学习路线

如果只想尽快跑通并理解主线，可以只读下面几个文件：

```text
README.md
configs/params.py
configs/datasets_config.py
data/dataset.py
network/face_sketch_net.py
training/train.py
training/retrieval.py
eval/metrics.py
train_val_test.py
```

## 建议理解顺序

```text
数据从哪里来
  -> Dataset 返回什么
  -> 模型输入输出是什么
  -> loss 如何组合
  -> 每轮训练如何更新
  -> 验证和测试如何计算检索指标
  -> research 脚本如何做消融实验
```
