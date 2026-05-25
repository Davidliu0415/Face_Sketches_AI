from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import datasets_config as data_config
from data.dataset import IMAGE_EXTENSIONS

MANIFEST_FIELDS = [
    "face_anchor",
    "sketch_positive",
    "identity",
    "dataset",
    "subset",
    "source",
    "photo_source",
    "sketch_source",
    "status",
]
MISSING_FIELDS = [
    "dataset",
    "subset",
    "identity",
    "photo_source",
    "sketch_source",
    "reason",
]

LFW_URLS = [
    "https://vis-www.cs.umass.edu/lfw/lfw.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
]
FGNET_URLS = [
    "https://yanweifu.github.io/FG_NET_data/FGNET.zip",
    "http://yanweifu.github.io/FG_NET_data/FGNET.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IIIT-D dataset03 test manifests.")
    parser.add_argument("--dataset-root", default=str(data_config.test_dataset03))
    parser.add_argument("--lfw-root", default="")
    parser.add_argument("--lfw-archive", default="")
    parser.add_argument("--fgnet-root", default="")
    parser.add_argument("--fgnet-archive", default="")
    parser.add_argument("--cufs-root", default="")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def clean_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "sample"


def read_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_format_extension(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image.verify()
            image_format = (image.format or "").upper()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Invalid image: {path}") from exc

    if image_format in {"JPEG", "JPG"}:
        return ".jpg"
    if image_format == "PNG":
        return ".png"
    if image_format == "BMP":
        return ".bmp"
    if image_format == "WEBP":
        return ".webp"
    suffix = path.suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"


def copy_verified_image(source: Path, destination_base: Path, force: bool = False) -> Path:
    suffix = image_format_extension(source)
    destination = destination_base.with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        image_format_extension(destination)
        return destination
    shutil.copy2(source, destination)
    image_format_extension(destination)
    return destination


def relative_to_manifest_root(path: Path, manifest_root: Path) -> str:
    return os.path.relpath(path, manifest_root).replace("\\", "/")


def build_stem_index(root: Path | None) -> dict[str, Path]:
    if root is None or not root.exists():
        return {}
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem.lower(), path)
    return index


def merge_indexes(*indexes: dict[str, Path]) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    for index in indexes:
        for key, path in index.items():
            merged.setdefault(key, path)
    return merged


def safe_extract_tar(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath([root, target]) != str(root):
                raise ValueError(f"Unsafe tar member path: {member.name}")
        handle.extractall(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            target = (destination / member).resolve()
            if os.path.commonpath([root, target]) != str(root):
                raise ValueError(f"Unsafe zip member path: {member}")
        handle.extractall(destination)


def extract_archive(archive: Path, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
        safe_extract_tar(archive, destination)
        return destination
    if archive.suffix.lower() == ".zip":
        safe_extract_zip(archive, destination)
        return destination
    raise ValueError(f"Unsupported archive format: {archive}")


def download_file(urls: list[str], destination: Path, force: bool = False) -> Path | None:
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                with destination.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            return destination
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
    if destination.exists():
        destination.unlink()
    if last_error is not None:
        print(f"Download failed for {destination.name}: {last_error}", flush=True)
    return None


def maybe_prepare_archive_source(
    root_value: str,
    archive_value: str,
    download_urls: list[str],
    archive_name: str,
    extract_dir: Path,
    no_download: bool,
    force: bool,
) -> Path | None:
    if root_value:
        root = Path(root_value)
        return root if root.exists() else None

    archive = Path(archive_value) if archive_value else None
    if archive is None and not no_download:
        archive = download_file(download_urls, extract_dir.parent / archive_name, force=force)
    if archive is None or not archive.exists():
        return None
    return extract_archive(archive, extract_dir, force=force)


def download_image(url: str, destination: Path, force: bool = False) -> Path | None:
    if destination.exists() and not force:
        try:
            image_format_extension(destination)
            return destination
        except ValueError:
            destination.unlink()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        image_format_extension(destination)
        return destination
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        print(f"Image download failed: {url} ({exc})", flush=True)
        if destination.exists():
            destination.unlink()
        return None


def exact_key(path: Path) -> str:
    return path.stem.lower()


def viewed_fgnet_key(path: Path) -> str:
    key = path.stem.lower()
    if re.match(r"^s\d{3}a", key):
        return key[1:]
    return key


def staff_key(path: Path) -> str:
    key = path.stem.lower()
    if len(key) > 1 and key[0] in {"p", "s"} and key[1:].isdigit():
        return key[1:]
    return key


class Dataset03Preparer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_root = Path(args.dataset_root)
        self.raw_root = self.dataset_root / "IIITD_SketchDatabase"
        self.processed_root = self.dataset_root / "processed"
        self.photos_root = self.processed_root / "photos"
        self.sketches_root = self.processed_root / "sketches"
        self.manifest_root = self.processed_root / "manifests"
        self.download_root = self.dataset_root / "external"
        self.rows_by_manifest: dict[str, list[dict[str, str]]] = {}
        self.missing_rows: list[dict[str, str]] = []

    def run(self) -> None:
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        lfw_root = maybe_prepare_archive_source(
            self.args.lfw_root,
            self.args.lfw_archive,
            LFW_URLS,
            "lfw.tgz",
            self.download_root / "lfw",
            no_download=self.args.no_download,
            force=self.args.force_download,
        )
        fgnet_root = maybe_prepare_archive_source(
            self.args.fgnet_root,
            self.args.fgnet_archive,
            FGNET_URLS,
            "FGNET.zip",
            self.download_root / "fgnet",
            no_download=self.args.no_download,
            force=self.args.force_download,
        )
        cufs_root = Path(self.args.cufs_root) if self.args.cufs_root else None
        if cufs_root is not None and not cufs_root.exists():
            cufs_root = None

        indexes = {
            "lfw": build_stem_index(lfw_root),
            "fgnet": build_stem_index(fgnet_root),
            "cufs": self.build_cuhk_index(cufs_root),
        }

        self.prepare_txt_subset(
            manifest_name="dataset03_viewed_fgnet",
            subset="viewed_fgnet",
            base_dir=self.raw_root / "Viewed sketch database" / "FG-NET",
            photo_index=indexes["fgnet"],
            sketch_key=viewed_fgnet_key,
            source="FG-NET",
        )
        self.prepare_txt_subset(
            manifest_name="dataset03_viewed_lfw",
            subset="viewed_lfw",
            base_dir=self.raw_root / "Viewed sketch database" / "LFW",
            photo_index=indexes["lfw"],
            sketch_key=exact_key,
            source="LFW",
        )
        self.prepare_directory_subset(
            manifest_name="dataset03_viewed_iiitd_staff",
            subset="viewed_iiitd_staff",
            base_dir=self.raw_root / "Viewed sketch database" / "IIIT-D student and staff",
            key_func=staff_key,
            source="IIIT-D student and staff",
        )
        self.prepare_txt_subset(
            manifest_name="dataset03_semi_cuhk",
            subset="semi_cuhk",
            base_dir=self.raw_root / "Semi-forensic database" / "CUHK",
            photo_index=indexes["cufs"],
            sketch_key=exact_key,
            source="CUHK",
        )
        self.prepare_txt_subset(
            manifest_name="dataset03_semi_fgnet",
            subset="semi_fgnet",
            base_dir=self.raw_root / "Semi-forensic database" / "FG-NET",
            photo_index=indexes["fgnet"],
            sketch_key=exact_key,
            source="FG-NET",
        )
        self.prepare_directory_subset(
            manifest_name="dataset03_semi_iiitd_staff",
            subset="semi_iiitd_staff",
            base_dir=self.raw_root / "Semi-forensic database" / "IIIT-D student and staff",
            key_func=staff_key,
            source="IIIT-D student and staff",
        )
        self.prepare_forensic_subset()
        self.write_outputs()

    def build_cuhk_index(self, cufs_root: Path | None) -> dict[str, Path]:
        local_indexes = []
        for path in [
            data_config.test_dataset02 / "archive" / "photos",
            data_config.test_dataset02 / "archive" / "photo",
        ]:
            local_indexes.append(build_stem_index(path))
        if cufs_root is not None:
            local_indexes.append(build_stem_index(cufs_root))
        index = merge_indexes(*local_indexes)
        for key, path in list(index.items()):
            match = re.match(r"^([fm])-(\d{3}-\d{2})$", key)
            if match:
                index.setdefault(f"{match.group(1)}1-{match.group(2)}", path)
        return index

    def prepare_txt_subset(
        self,
        manifest_name: str,
        subset: str,
        base_dir: Path,
        photo_index: dict[str, Path],
        sketch_key: Callable[[Path], str],
        source: str,
    ) -> None:
        names = read_names(base_dir / "photo" / "photo.txt")
        sketches = {
            sketch_key(path): path
            for path in sorted((base_dir / "sketch").glob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        rows: list[dict[str, str]] = []
        for name in names:
            key = name.lower()
            photo_path = photo_index.get(key)
            sketch_path = sketches.get(key)
            identity = f"{subset}:{name}"
            if photo_path is None or sketch_path is None:
                self.add_missing(
                    subset,
                    identity,
                    str(photo_path or name),
                    str(sketch_path or ""),
                    "missing_photo" if photo_path is None else "missing_sketch",
                )
                continue
            row = self.add_pair(subset, identity, photo_path, sketch_path, source, "IIIT-D sketch")
            if row is not None:
                rows.append(row)
        self.rows_by_manifest[manifest_name] = rows

    def prepare_directory_subset(
        self,
        manifest_name: str,
        subset: str,
        base_dir: Path,
        key_func: Callable[[Path], str],
        source: str,
    ) -> None:
        photo_dir = base_dir / "photo"
        sketch_dir = base_dir / "sketch"
        photos = {
            key_func(path): path
            for path in sorted(photo_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        sketches = {
            key_func(path): path
            for path in sorted(sketch_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        rows: list[dict[str, str]] = []
        for key, photo_path in photos.items():
            sketch_path = sketches.get(key)
            identity = f"{subset}:{key}"
            if sketch_path is None:
                self.add_missing(subset, identity, str(photo_path), "", "missing_sketch")
                continue
            row = self.add_pair(subset, identity, photo_path, sketch_path, source, "IIIT-D sketch")
            if row is not None:
                rows.append(row)
        for key, sketch_path in sketches.items():
            if key not in photos:
                self.add_missing(subset, f"{subset}:{key}", "", str(sketch_path), "missing_photo")
        self.rows_by_manifest[manifest_name] = rows

    def prepare_forensic_subset(self) -> None:
        subset = "forensic_verified"
        manifest_name = "dataset03_forensic_verified"
        rows: list[dict[str, str]] = []
        source_path = self.raw_root / "Forensic sketch database" / "Forensic_sketches.txt"
        if not source_path.exists():
            self.rows_by_manifest[manifest_name] = rows
            return

        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                urls = [part.strip() for part in line.split() if part.startswith(("http://", "https://"))]
                identity = f"{subset}:line_{line_number:03d}"
                if len(urls) != 2:
                    if urls:
                        self.add_missing(subset, identity, "", " ".join(urls), "skipped_unpaired_forensic_url")
                    continue
                if self.args.no_download:
                    self.add_missing(subset, identity, urls[1], urls[0], "download_disabled")
                    continue
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_root = Path(tmpdir)
                    sketch_path = download_image(
                        urls[0],
                        tmp_root / f"forensic_{line_number:03d}_sketch.img",
                        force=self.args.force_download,
                    )
                    photo_path = download_image(
                        urls[1],
                        tmp_root / f"forensic_{line_number:03d}_photo.img",
                        force=self.args.force_download,
                    )
                    if photo_path is None or sketch_path is None:
                        self.add_missing(subset, identity, urls[1], urls[0], "download_or_validation_failed")
                        continue
                    row = self.add_pair(subset, identity, photo_path, sketch_path, urls[1], urls[0])
                    if row is not None:
                        rows.append(row)
        self.rows_by_manifest[manifest_name] = rows

    def add_pair(
        self,
        subset: str,
        identity: str,
        photo_path: Path,
        sketch_path: Path,
        photo_source: str,
        sketch_source: str,
    ) -> dict[str, str] | None:
        destination_name = f"dataset03_{subset}__{clean_name(identity)}"
        try:
            photo_dest = copy_verified_image(
                photo_path,
                self.photos_root / subset / destination_name,
                force=self.args.force_download,
            )
            sketch_dest = copy_verified_image(
                sketch_path,
                self.sketches_root / subset / destination_name,
                force=self.args.force_download,
            )
        except ValueError as exc:
            self.add_missing(subset, identity, str(photo_path), str(sketch_path), str(exc))
            return None
        return {
            "face_anchor": relative_to_manifest_root(photo_dest, self.manifest_root),
            "sketch_positive": relative_to_manifest_root(sketch_dest, self.manifest_root),
            "identity": identity,
            "dataset": "dataset03",
            "subset": subset,
            "source": "IIIT-D Sketch Database",
            "photo_source": str(photo_source),
            "sketch_source": str(sketch_source),
            "status": "ok",
        }

    def add_missing(
        self,
        subset: str,
        identity: str,
        photo_source: str,
        sketch_source: str,
        reason: str,
    ) -> None:
        self.missing_rows.append(
            {
                "dataset": "dataset03",
                "subset": subset,
                "identity": identity,
                "photo_source": photo_source,
                "sketch_source": sketch_source,
                "reason": reason,
            }
        )

    def write_outputs(self) -> None:
        all_rows: list[dict[str, str]] = []
        for manifest_name, rows in self.rows_by_manifest.items():
            all_rows.extend(rows)
            self.write_manifest(self.manifest_root / f"{manifest_name}.csv", rows)
        self.write_manifest(self.manifest_root / "dataset03_all.csv", all_rows)
        self.write_missing(self.manifest_root / "missing_dataset03.csv")
        print(f"Prepared dataset03 samples: {len(all_rows)}", flush=True)
        for manifest_name, rows in sorted(self.rows_by_manifest.items()):
            print(f"  {manifest_name}: {len(rows)}", flush=True)
        print(f"Missing or skipped samples: {len(self.missing_rows)}", flush=True)
        print(f"Manifest root: {self.manifest_root}", flush=True)

    def write_manifest(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def write_missing(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MISSING_FIELDS)
            writer.writeheader()
            writer.writerows(self.missing_rows)


def main() -> None:
    args = parse_args()
    Dataset03Preparer(args).run()


if __name__ == "__main__":
    main()
