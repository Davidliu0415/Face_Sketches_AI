from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import datasets_config as data_config
from configs import params
from data.dataset import manifest_identity, manifest_negative_identity


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val manifests with disjoint identity sets.")
    parser.add_argument(
        "--input-manifest",
        action="append",
        default=None,
        help="Input triplet manifest. Pass multiple times to merge sources before splitting.",
    )
    parser.add_argument("--train-output", default=str(data_config.unseen_train_manifest))
    parser.add_argument("--val-output", default=str(data_config.unseen_val_manifest))
    parser.add_argument("--summary-output", default=str(data_config.data_root / "unseen_identity_split_summary.json"))
    parser.add_argument("--val-identity-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=params.seed)
    return parser.parse_args()


def read_rows(paths: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    return rows, fieldnames


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_fieldnames(fieldnames: list[str]) -> list[str]:
    output = list(fieldnames)
    for name in ("sketch_negative", "sketch_negative_label", "negative_identity"):
        if name not in output:
            output.append(name)
    return output


def build_sketch_pool(rows: list[dict[str, str]], allowed_identities: set[str]) -> dict[str, list[str]]:
    sketches_by_identity: dict[str, list[str]] = {}
    for row in rows:
        identity = manifest_identity(row)
        if identity not in allowed_identities:
            continue
        sketch_path = row.get("sketch_positive", "").strip()
        if not sketch_path:
            continue
        sketches_by_identity.setdefault(identity, []).append(sketch_path)
    return sketches_by_identity


def choose_negative_identity(
    identity_pool: list[str],
    identity_index: int | None,
    rng: random.Random,
) -> str | None:
    if not identity_pool or (identity_index is not None and len(identity_pool) < 2):
        return None

    if identity_index is None:
        return rng.choice(identity_pool)

    index = rng.randrange(len(identity_pool) - 1)
    if index >= identity_index:
        index += 1
    return identity_pool[index]


def resample_split_rows(
    rows: list[dict[str, str]],
    allowed_identities: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    sketches_by_identity = build_sketch_pool(rows, allowed_identities)
    negative_identity_pool = sorted(sketches_by_identity)
    negative_identity_index = {identity: index for index, identity in enumerate(negative_identity_pool)}
    output_rows: list[dict[str, str]] = []
    stats = {
        "candidate_rows": 0,
        "retained_rows": 0,
        "resampled_negative_rows": 0,
        "missing_negative_pool_rows": 0,
        "original_negative_in_split_rows": 0,
        "original_negative_cross_split_rows": 0,
        "negative_identity_violations": 0,
    }

    for row in rows:
        identity = manifest_identity(row)
        if identity not in allowed_identities:
            continue

        stats["candidate_rows"] += 1
        original_negative_identity = manifest_negative_identity(row)
        if original_negative_identity in allowed_identities and original_negative_identity != identity:
            stats["original_negative_in_split_rows"] += 1
        else:
            stats["original_negative_cross_split_rows"] += 1

        negative_identity = choose_negative_identity(
            negative_identity_pool,
            negative_identity_index.get(identity),
            rng,
        )
        if negative_identity is None:
            stats["missing_negative_pool_rows"] += 1
            continue

        negative_sketch = rng.choice(sketches_by_identity[negative_identity])
        if negative_identity == identity:
            stats["negative_identity_violations"] += 1
            continue

        updated = dict(row)
        updated["sketch_negative"] = negative_sketch
        updated["sketch_negative_label"] = "0"
        updated["negative_identity"] = negative_identity
        output_rows.append(updated)
        stats["retained_rows"] += 1
        stats["resampled_negative_rows"] += 1

    return output_rows, stats


def main():
    args = parse_args()
    input_manifests = args.input_manifest or [str(data_config.train_manifest), str(data_config.val_manifest)]
    rows, fieldnames = read_rows(input_manifests)
    identities = sorted({manifest_identity(row) for row in rows})
    fieldnames = ensure_fieldnames(fieldnames)
    rng = random.Random(args.seed)
    rng.shuffle(identities)

    val_count = max(1, round(len(identities) * args.val_identity_ratio))
    val_identities = set(identities[:val_count])
    train_identities = set(identities[val_count:])

    train_rows, train_stats = resample_split_rows(rows, train_identities, rng)
    val_rows, val_stats = resample_split_rows(rows, val_identities, rng)

    write_manifest(Path(args.train_output), fieldnames, train_rows)
    write_manifest(Path(args.val_output), fieldnames, val_rows)

    identity_overlap = len(train_identities.intersection(val_identities))
    summary = {
        "input_rows": len(rows),
        "input_manifests": input_manifests,
        "seed": args.seed,
        "val_identity_ratio": args.val_identity_ratio,
        "train_identities": len(train_identities),
        "val_identities": len(val_identities),
        "identity_overlap": identity_overlap,
        "discarded_rows": len(rows) - len(train_rows) - len(val_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_output": str(Path(args.train_output)),
        "val_output": str(Path(args.val_output)),
    }
    with Path(args.summary_output).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
