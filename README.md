# Face Sketch Matcher

This project follows the layout of `gc2sa_net`: configuration lives in
`configs`, dataset code in `data`, model and losses in `network`, and the
training entry point in `training`.

## Architecture

- Backbone: shared `MobileFaceNet` encoder for face and sketch images.
- Embedding: L2-normalized feature vector, default dimension `512`.
- Heads: two independent ArcFace classification heads:
  - `face_head`
  - `sketch_head`
- Matching loss: triplet loss over `(face_anchor, sketch_positive, sketch_negative)`.

The ArcFace heads default to two output classes because the project is set up
for binary face/sketch matching first. You can change
`face_num_classes` and `sketch_num_classes` in `configs/params.py` when the
real labels are finalized.

## Project Tree

```text
configs/
  datasets_config.py
  params.py
data/
  dataset.py
network/
  face_sketch_net.py
  heads.py
  losses.py
  mobilefacenet.py
training/
  main.py
  train.py
eval/
  metrics.py
```

## Dataset Placeholder

No dataset is included yet. `data/dataset.py` only defines the expected
interface. A future CSV manifest can use these columns:

```csv
face_anchor,sketch_positive,sketch_negative,face_binary_label,sketch_binary_label,sketch_negative_label
```

Relative paths are resolved from `configs/datasets_config.py:image_root`.

## Quick Checks

Run a synthetic forward/backward pass without data:

```bash
python training/main.py --dry-run
```

Start real training after a manifest exists:

```bash
python training/main.py --manifest data/train_manifest.csv
```
