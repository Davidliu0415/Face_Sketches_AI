from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DefaultTransform:
    def __init__(self, image_size: int = 112) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((self.image_size, self.image_size), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5


def default_transform(image_size: int = 112) -> Callable:
    return DefaultTransform(image_size=image_size)


def load_rgb_image(path: str | Path, transform: Callable) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return transform(image)


def image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    return sorted(file for file in root.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def normalized_pair_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in ("-sz1", "_sz1", "-sketch", "_sketch", "-photo", "_photo"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    key = re.sub(r"^([fm])2-", r"\1-", key)
    return key


def manifest_identity(row: Mapping[str, str]) -> str:
    identity = row.get("identity", "").strip()
    if identity:
        return identity
    return Path(row["face_anchor"]).parent.name or normalized_pair_key(Path(row["face_anchor"]))


def manifest_negative_identity(row: Mapping[str, str]) -> str:
    identity = row.get("negative_identity", "").strip()
    if identity:
        return identity
    return Path(row["sketch_negative"]).parent.name or normalized_pair_key(Path(row["sketch_negative"]))


def build_identity_label_map(
    manifest_paths: str | Path | list[str | Path],
) -> dict[str, int]:
    if not isinstance(manifest_paths, list):
        manifest_paths = [manifest_paths]

    identities: set[str] = set()
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                identities.add(manifest_identity(row))
                if row.get("sketch_negative"):
                    identities.add(manifest_negative_identity(row))

    if not identities:
        raise ValueError(f"No identities found in manifests: {manifest_paths}")

    return {identity: index for index, identity in enumerate(sorted(identities))}


class FaceSketchTripletDataset(Dataset):
    """CSV-backed placeholder dataset for face/sketch matching.

    Expected columns:
    face_anchor, sketch_positive, sketch_negative

    Optional columns:
    face_binary_label, sketch_binary_label, sketch_negative_label
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        transform: Callable | None = None,
        image_size: int = 112,
        identity_to_label: Mapping[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root is not None else self.manifest_path.parent
        self.transform = transform or default_transform(image_size)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                "Create it later or run training/main.py with --dry-run."
            )

        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.samples = list(reader)

        required = {"face_anchor", "sketch_positive", "sketch_negative"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        self.identity_to_label = (
            dict(identity_to_label)
            if identity_to_label is not None
            else build_identity_label_map(self.manifest_path)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.samples[index]

        face_anchor = self._load_image(row["face_anchor"])
        sketch_positive = self._load_image(row["sketch_positive"])
        sketch_negative = self._load_image(row["sketch_negative"])
        identity = manifest_identity(row)
        negative_identity = manifest_negative_identity(row)

        return {
            "face_anchor": face_anchor,
            "sketch_positive": sketch_positive,
            "sketch_negative": sketch_negative,
            "face_identity_label": torch.tensor(self._identity_label(identity), dtype=torch.long),
            "sketch_identity_label": torch.tensor(self._identity_label(identity), dtype=torch.long),
            "sketch_negative_identity_label": torch.tensor(self._identity_label(negative_identity), dtype=torch.long),
            "face_binary_label": torch.tensor(int(row.get("face_binary_label", 1)), dtype=torch.long),
            "sketch_binary_label": torch.tensor(int(row.get("sketch_binary_label", 1)), dtype=torch.long),
            "sketch_negative_label": torch.tensor(int(row.get("sketch_negative_label", 0)), dtype=torch.long),
        }

    def _load_image(self, value: str) -> torch.Tensor:
        path = Path(value)
        if not path.is_absolute():
            path = self.image_root / path

        return load_rgb_image(path, self.transform)

    def _identity_label(self, identity: str) -> int:
        try:
            return self.identity_to_label[identity]
        except KeyError as exc:
            raise KeyError(f"Identity {identity!r} is missing from the label map.") from exc


class FaceSketchPairDataset(Dataset):
    """Paired face/sketch dataset for retrieval-style validation or testing."""

    def __init__(
        self,
        samples: list[tuple[Path, Path, str]],
        transform: Callable | None = None,
        image_size: int = 112,
        identity_to_label: Mapping[str, int] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("FaceSketchPairDataset received no samples.")
        self.samples = samples
        self.transform = transform or default_transform(image_size)
        self.identity_to_label = dict(identity_to_label) if identity_to_label is not None else None

    @classmethod
    def from_directories(
        cls,
        face_dir: str | Path,
        sketch_dir: str | Path,
        transform: Callable | None = None,
        image_size: int = 112,
        identity_to_label: Mapping[str, int] | None = None,
    ) -> "FaceSketchPairDataset":
        face_files = image_files(face_dir)
        sketch_files = image_files(sketch_dir)
        sketch_by_key = {normalized_pair_key(path): path for path in sketch_files}

        samples = []
        for face_path in face_files:
            key = normalized_pair_key(face_path)
            sketch_path = sketch_by_key.get(key)
            if sketch_path is not None:
                samples.append((face_path, sketch_path, key))

        if not samples and len(face_files) == len(sketch_files):
            samples = [
                (face_path, sketch_path, str(index))
                for index, (face_path, sketch_path) in enumerate(zip(face_files, sketch_files))
            ]

        return cls(samples=samples, transform=transform, image_size=image_size, identity_to_label=identity_to_label)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        transform: Callable | None = None,
        image_size: int = 112,
        identity_to_label: Mapping[str, int] | None = None,
    ) -> "FaceSketchPairDataset":
        manifest = Path(manifest_path)
        root = Path(image_root) if image_root is not None else manifest.parent
        samples = []

        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"face_anchor", "sketch_positive"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

            for row in reader:
                face_path = Path(row["face_anchor"])
                sketch_path = Path(row["sketch_positive"])
                if not face_path.is_absolute():
                    face_path = root / face_path
                if not sketch_path.is_absolute():
                    sketch_path = root / sketch_path
                samples.append((face_path, sketch_path, manifest_identity(row)))

        return cls(samples=samples, transform=transform, image_size=image_size, identity_to_label=identity_to_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        face_path, sketch_path, identity = self.samples[index]
        item = {
            "face": load_rgb_image(face_path, self.transform),
            "sketch": load_rgb_image(sketch_path, self.transform),
            "identity": identity,
            "face_path": str(face_path),
            "sketch_path": str(sketch_path),
        }
        if self.identity_to_label is not None:
            item["identity_label"] = torch.tensor(self.identity_to_label[identity], dtype=torch.long)
        return item


def load_pair_dataset_from_test_set(
    test_set: Mapping[str, object],
    transform: Callable | None = None,
    image_size: int = 112,
    identity_to_label: Mapping[str, int] | None = None,
) -> FaceSketchPairDataset:
    kind = str(test_set.get("kind", "directories"))
    if kind == "manifest":
        image_root = test_set.get("image_root")
        return FaceSketchPairDataset.from_manifest(
            manifest_path=Path(str(test_set["manifest"])),
            image_root=Path(str(image_root)) if image_root else None,
            transform=transform,
            image_size=image_size,
            identity_to_label=identity_to_label,
        )
    if kind == "directories":
        return FaceSketchPairDataset.from_directories(
            face_dir=Path(str(test_set["face_dir"])),
            sketch_dir=Path(str(test_set["sketch_dir"])),
            transform=transform,
            image_size=image_size,
            identity_to_label=identity_to_label,
        )
    raise ValueError(f"Unsupported test set kind: {kind}")
