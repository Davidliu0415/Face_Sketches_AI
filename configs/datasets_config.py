import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_root = project_root / "data"
image_root = data_root / "images" / "structured_face_sketch_dataset"

train_manifest = data_root / "train_manifest.csv"
val_manifest = data_root / "val_manifest.csv"
unseen_train_manifest = data_root / "train_manifest_unseen_identity.csv"
unseen_val_manifest = data_root / "val_manifest_unseen_identity.csv"

pairs_manifest = image_root / "manifests" / "face_sketch_pairs.csv"
source_manifest = image_root / "manifests" / "face_sketch_manifest.csv"

test_root = data_root / "test"
test_dataset01 = test_root / "dataset01"
test_dataset02 = test_root / "dataset02"
test_dataset03 = test_root / "dataset03"
test_dataset03_processed = test_dataset03 / "processed"
test_dataset03_manifest_root = test_dataset03_processed / "manifests"


def directory_test_set(name: str, face_dir: Path, sketch_dir: Path) -> dict:
    return {
        "name": name,
        "kind": "directories",
        "face_dir": face_dir,
        "sketch_dir": sketch_dir,
    }


def manifest_test_set(name: str, manifest: Path) -> dict:
    return {
        "name": name,
        "kind": "manifest",
        "manifest": manifest,
    }


TEST_SETS = [
    directory_test_set(
        "dataset01",
        test_dataset01 / "archive" / "photos",
        test_dataset01 / "archive" / "sketches",
    ),
    directory_test_set(
        "dataset02",
        test_dataset02 / "archive" / "photos",
        test_dataset02 / "archive" / "sketches",
    ),
    manifest_test_set("dataset03_all", test_dataset03_manifest_root / "dataset03_all.csv"),
    manifest_test_set("dataset03_viewed_fgnet", test_dataset03_manifest_root / "dataset03_viewed_fgnet.csv"),
    manifest_test_set("dataset03_viewed_lfw", test_dataset03_manifest_root / "dataset03_viewed_lfw.csv"),
    manifest_test_set(
        "dataset03_viewed_iiitd_staff",
        test_dataset03_manifest_root / "dataset03_viewed_iiitd_staff.csv",
    ),
    manifest_test_set("dataset03_semi_cuhk", test_dataset03_manifest_root / "dataset03_semi_cuhk.csv"),
    manifest_test_set("dataset03_semi_fgnet", test_dataset03_manifest_root / "dataset03_semi_fgnet.csv"),
    manifest_test_set(
        "dataset03_semi_iiitd_staff",
        test_dataset03_manifest_root / "dataset03_semi_iiitd_staff.csv",
    ),
    manifest_test_set(
        "dataset03_forensic_verified",
        test_dataset03_manifest_root / "dataset03_forensic_verified.csv",
    ),
]


def _manifest_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return next(reader, None) is not None


def test_set_available(test_set: dict) -> bool:
    if test_set["kind"] == "manifest":
        return _manifest_has_rows(Path(test_set["manifest"]))
    return Path(test_set["face_dir"]).exists() and Path(test_set["sketch_dir"]).exists()


def get_test_sets(include_unavailable: bool = False) -> list[dict]:
    if include_unavailable:
        return [dict(test_set) for test_set in TEST_SETS]
    return [dict(test_set) for test_set in TEST_SETS if test_set_available(test_set)]
