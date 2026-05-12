from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from network.losses import TripletLoss, classification_accuracy


@dataclass
class TrainStats:
    loss: float
    triplet_loss: float
    face_ce_loss: float
    sketch_ce_loss: float
    face_acc: float
    sketch_acc: float


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    triplet_loss: TripletLoss,
    triplet_weight=1.0,
    face_ce_weight=1.0,
    sketch_ce_weight=1.0,
    show_progress=True,
):
    model.train()
    totals = {
        "loss": 0.0,
        "triplet_loss": 0.0,
        "face_ce_loss": 0.0,
        "sketch_ce_loss": 0.0,
        "face_acc": 0.0,
        "sketch_acc": 0.0,
    }
    steps = 0

    iterator = tqdm(data_loader, desc="Train", leave=False) if show_progress and tqdm is not None else data_loader
    for batch in iterator:
        face_anchor = batch["face_anchor"].to(device)
        sketch_positive = batch["sketch_positive"].to(device)
        sketch_negative = batch["sketch_negative"].to(device)

        face_embedding = model.embed(face_anchor)
        sketch_positive_embedding = model.embed(sketch_positive)
        sketch_negative_embedding = model.embed(sketch_negative)

        loss_triplet = triplet_loss(face_embedding, sketch_positive_embedding, sketch_negative_embedding)
        loss = triplet_weight * loss_triplet

        face_ce_loss = torch.zeros((), device=device)
        sketch_ce_loss = torch.zeros((), device=device)
        face_acc = torch.zeros((), device=device)
        sketch_acc = torch.zeros((), device=device)

        if "face_binary_label" in batch and face_ce_weight > 0:
            face_labels = batch["face_binary_label"].to(device)
            face_logits = model.face_head(face_embedding, face_labels)
            face_ce_loss = F.cross_entropy(face_logits, face_labels)
            face_acc = classification_accuracy(face_logits, face_labels)
            loss = loss + face_ce_weight * face_ce_loss

        if "sketch_binary_label" in batch and sketch_ce_weight > 0:
            sketch_labels = batch["sketch_binary_label"].to(device)
            sketch_embeddings = sketch_positive_embedding

            if "sketch_negative_label" in batch:
                sketch_negative_labels = batch["sketch_negative_label"].to(device)
                sketch_embeddings = torch.cat([sketch_positive_embedding, sketch_negative_embedding], dim=0)
                sketch_labels = torch.cat([sketch_labels, sketch_negative_labels], dim=0)

            sketch_logits = model.sketch_head(sketch_embeddings, sketch_labels)
            sketch_ce_loss = F.cross_entropy(sketch_logits, sketch_labels)
            sketch_acc = classification_accuracy(sketch_logits, sketch_labels)
            loss = loss + sketch_ce_weight * sketch_ce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        steps += 1
        totals["loss"] += loss.item()
        totals["triplet_loss"] += loss_triplet.item()
        totals["face_ce_loss"] += face_ce_loss.item()
        totals["sketch_ce_loss"] += sketch_ce_loss.item()
        totals["face_acc"] += face_acc.item()
        totals["sketch_acc"] += sketch_acc.item()

    if steps == 0:
        raise ValueError("The training dataloader produced no batches.")

    return TrainStats(**{key: value / steps for key, value in totals.items()})
