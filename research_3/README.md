# research_3: Face-Sketch ArcFace + TripletLoss

This folder contains the stage-3 experiment from `docs/research_plan.md`.

The experiment uses the shared MobileFaceNet encoder, the existing face/sketch
ArcFace heads, and TripletLoss together:

```text
triplet_weight = 1.0
face_ce_weight = 1.0
sketch_ce_weight = 1.0
```

TripletLoss uses face images as anchors, matched sketches as positives, and
unmatched sketches as negatives. The ArcFace heads keep the current project
binary label scheme for face and sketch branches.

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
python research_3\run_arcface_triplet.py --epochs 1 --limit-train 64 --limit-val 64 --limit-test 64 --batch-size 16 --eval-batch-size 32 --no-amp
```

If the Windows `python` command points to the Microsoft Store placeholder, use
the bundled Codex Python executable shown by the workspace dependencies tool.

## Full ArcFace + Triplet Run

```powershell
python research_3\run_arcface_triplet.py --epochs 30 --batch-size 64 --eval-batch-size 128 --amp
```

If GPU memory is insufficient, use the same experiment with smaller batches:

```powershell
python research_3\run_arcface_triplet.py --epochs 30 --batch-size 32 --eval-batch-size 64 --amp
```

## Outputs

Each run directory is written under:

```text
checkpoints/research_3/arcface_triplet_001_YYYYMMDD_HHMMSS/
```

It contains:

```text
best.pth
last.pth
metrics.csv
arcface_triplet_results.csv
failed_matches.csv
```

- `metrics.csv` records train/val loss, TripletLoss, ArcFace CE losses,
  triplet accuracy, and face/sketch ArcFace accuracies.
- `arcface_triplet_results.csv` records dataset sample counts, Top-1, Top-5,
  mean positive cosine, runtime, and checkpoint path.
- `failed_matches.csv` records Top-1 retrieval failures with face image,
  predicted sketch, correct sketch, similarity, correct similarity, rank, and
  blank error-type fields for later manual analysis.

## Stage Comparison Template

Fill this table from the three stage result CSV files:

```markdown
| Stage | Run ID | dataset01 Top-1 | dataset01 Top-5 | dataset02 Top-1 | dataset02 Top-5 | Average Top-1 | Average Top-5 | Conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | `baseline_001` |  |  |  |  |  |  |  |
| ArcFace Only | `arcface_only_001` |  |  |  |  |  |  |  |
| ArcFace + TripletLoss | `arcface_triplet_001` |  |  |  |  |  |  |  |
```

Use:

```text
checkpoints/research_1/*/baseline_results.csv
checkpoints/research_2/*/arcface_only_results.csv
checkpoints/research_3/*/arcface_triplet_results.csv
```

## Stage Boundary

This stage intentionally does not change the project-wide config, model
structure, or label scheme. It is the planned joint-objective experiment for
comparing against the stage-1 Triplet baseline and the stage-2 ArcFace-only
ablation.
