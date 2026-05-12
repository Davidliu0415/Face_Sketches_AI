from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.samples[index]

        face_anchor = self._load_image(row["face_anchor"])
        sketch_positive = self._load_image(row["sketch_positive"])
        sketch_negative = self._load_image(row["sketch_negative"])

        return {
            "face_anchor": face_anchor,
            "sketch_positive": sketch_positive,
            "sketch_negative": sketch_negative,
            "face_binary_label": torch.tensor(int(row.get("face_binary_label", 1)), dtype=torch.long),
            "sketch_binary_label": torch.tensor(int(row.get("sketch_binary_label", 1)), dtype=torch.long),
            "sketch_negative_label": torch.tensor(int(row.get("sketch_negative_label", 0)), dtype=torch.long),
        }

    def _load_image(self, value: str) -> torch.Tensor:
        path = Path(value)
        if not path.is_absolute():
            path = self.image_root / path

        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)
