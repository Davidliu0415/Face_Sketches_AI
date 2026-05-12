from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_root = project_root / "data"
image_root = data_root / "images"

train_manifest = data_root / "train_manifest.csv"
val_manifest = data_root / "val_manifest.csv"
