# research_2: Face-Sketch ArcFace Only

This folder contains the independent stage-2 experiment from
`docs/research_plan.md`.

The experiment keeps the shared MobileFaceNet encoder and the existing
face/sketch ArcFace heads, but disables TripletLoss as a training objective:

```text
triplet_weight = 0.0
face_ce_weight = 1.0
sketch_ce_weight = 1.0
```

Triplet loss and triplet accuracy are still logged as observation metrics, but
they do not contribute to the training loss in this stage.

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

```powershell
& "C:\Users\liuzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" research_2\run_arcface_only.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32 --no-amp
```

## Full ArcFace-Only Run

```powershell
& "C:\Users\liuzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" research_2\run_arcface_only.py --epochs 30 --batch-size 64 --eval-batch-size 128 --amp
```

If GPU memory is insufficient, use the same experiment with smaller batches:

```powershell
& "C:\Users\liuzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" research_2\run_arcface_only.py --epochs 30 --batch-size 32 --eval-batch-size 64 --amp
```

## Outputs

Each run directory is written under:

```text
checkpoints/research_2/arcface_only_001_YYYYMMDD_HHMMSS/
```

It contains:

```text
best.pth
last.pth
metrics.csv
arcface_only_results.csv
failed_matches.csv
```

- `metrics.csv` records train/val loss, ArcFace CE losses, ArcFace
  accuracies, and triplet observation metrics.
- `arcface_only_results.csv` records dataset sample counts, Top-1, Top-5,
  mean positive cosine, runtime, and checkpoint path.
- `failed_matches.csv` records Top-1 retrieval failures with face image,
  predicted sketch, correct sketch, similarity, correct similarity, rank, and
  blank error-type fields for later manual analysis.

## Stage Boundary

This stage intentionally does not change the project-wide config, model
structure, or identity-label scheme. It tests the planned ArcFace-only
ablation before any identity multi-class ArcFace follow-up experiment.
