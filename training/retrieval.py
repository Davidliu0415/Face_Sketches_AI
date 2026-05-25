from __future__ import annotations

import csv
from pathlib import Path

import torch

from eval.metrics import retrieval_diagnostics, retrieval_failures, retrieval_metrics
from training.train import autocast_context


@torch.no_grad()
def embed_pair_dataset(model, data_loader, device, use_amp=False):
    model.eval()
    face_embeddings = []
    sketch_embeddings = []
    face_paths = []
    sketch_paths = []
    identities = []
    identity_labels = []

    for batch in data_loader:
        face = batch["face"].to(device, non_blocking=True)
        sketch = batch["sketch"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            face_embeddings.append(model.embed(face).float().cpu())
            sketch_embeddings.append(model.embed(sketch).float().cpu())
        face_paths.extend(list(batch["face_path"]))
        sketch_paths.extend(list(batch["sketch_path"]))
        identities.extend(list(batch["identity"]))
        if "identity_label" in batch:
            identity_labels.extend(batch["identity_label"].tolist())

    labels = torch.tensor(identity_labels, dtype=torch.long) if identity_labels else None
    return (
        torch.cat(face_embeddings, dim=0),
        torch.cat(sketch_embeddings, dim=0),
        face_paths,
        sketch_paths,
        identities,
        labels,
    )


def retrieval_report(
    model,
    data_loader,
    device,
    use_amp,
    run_id,
    dataset_name,
    include_failures=True,
):
    face_embeddings, sketch_embeddings, face_paths, sketch_paths, identities, labels = embed_pair_dataset(
        model,
        data_loader,
        device,
        use_amp=use_amp,
    )
    metrics = retrieval_metrics(
        face_embeddings,
        sketch_embeddings,
        face_labels=labels,
        sketch_labels=labels,
    )
    diagnostics = retrieval_diagnostics(face_embeddings, sketch_embeddings, sketch_paths=sketch_paths)
    failures = []
    if include_failures:
        failures = retrieval_failures(
            face_embeddings=face_embeddings,
            sketch_embeddings=sketch_embeddings,
            face_paths=face_paths,
            sketch_paths=sketch_paths,
            identities=identities,
            run_id=run_id,
            dataset_name=dataset_name,
            face_labels=labels,
            sketch_labels=labels,
        )
    return metrics, diagnostics, failures


def write_diagnostics_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["run_id", "dataset", "sketch_index", "count", "share", "sketch_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
