# research_1: Face-Sketch Baseline

This folder contains the stage-1 baseline from `docs/research_plan.md`.

The baseline uses the existing project datasets and trains only the shared
embedding space with TripletLoss:

```text
triplet_weight = 1.0
face_ce_weight = 0.0
sketch_ce_weight = 0.0
```

ArcFace heads remain part of the model definition for checkpoint compatibility,
but their classification losses are disabled for this stage.

## Data

The script reads the existing project paths:

```text
data/train_manifest.csv
data/val_manifest.csv
data/images/structured_face_sketch_dataset
data/test/dataset01
data/test/dataset02
```

## Sanity Check

```bash
python research_1/run_baseline.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32 --no-amp
```

If the Windows `python` command points to the Microsoft Store placeholder, use
the bundled Codex Python executable shown by the workspace dependencies tool.

## Full Baseline Run

```bash
python research_1/run_baseline.py --epochs 30 --batch-size 64 --eval-batch-size 128 --amp
```

The default run id is `baseline_001`, and outputs are written under:

```text
checkpoints/research_1/baseline_001_YYYYMMDD_HHMMSS/
```

## Outputs

Each run directory contains:

```text
best.pth
last.pth
metrics.csv
baseline_results.csv
failed_matches.csv
```

- `metrics.csv` records train/val loss, triplet loss, and triplet accuracy.
- `baseline_results.csv` records dataset sample counts, Top-1, Top-5,
  mean positive cosine, runtime, and checkpoint path.
- `failed_matches.csv` records Top-1 retrieval failures with face image,
  predicted sketch, correct sketch, similarity, correct similarity, rank, and
  blank error-type fields for later manual analysis.
