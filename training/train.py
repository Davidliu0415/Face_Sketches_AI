from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.nn import functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from network.losses import BatchHardTripletLoss, TripletLoss, classification_accuracy


def autocast_context(device, enabled=False):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


@dataclass
class TrainStats:
    loss: float
    triplet_loss: float
    batch_hard_loss: float
    face_ce_loss: float
    sketch_ce_loss: float
    face_acc: float
    sketch_acc: float
    triplet_acc: float = 0.0
    grad_norm: float = 0.0
    nonfinite_batches: int = 0


def _batch_hard_inputs(batch, face_embedding, sketch_positive_embedding, sketch_negative_embedding, device):
    required = {"face_identity_label", "sketch_identity_label", "sketch_negative_identity_label"}
    if not required.issubset(batch):
        return None, None

    embeddings = torch.cat([face_embedding, sketch_positive_embedding, sketch_negative_embedding], dim=0)
    labels = torch.cat(
        [
            batch["face_identity_label"].to(device, non_blocking=True),
            batch["sketch_identity_label"].to(device, non_blocking=True),
            batch["sketch_negative_identity_label"].to(device, non_blocking=True),
        ],
        dim=0,
    )
    return embeddings, labels


def _grad_norm(parameters):
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().data.norm(2)
        value = parameter_norm * parameter_norm
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    return torch.sqrt(squared_norm).item()


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    triplet_loss: TripletLoss,
    batch_hard_loss: BatchHardTripletLoss | None = None,
    triplet_weight=1.0,
    batch_hard_weight=0.0,
    face_ce_weight=1.0,
    sketch_ce_weight=1.0,
    show_progress=True,
    log_interval=100,
    accumulation_steps=1,
    use_amp=False,
    scaler=None,
    track_grad_norm=False,
):
    model.train()
    accumulation_steps = max(1, accumulation_steps)
    totals = {
        "loss": 0.0,
        "triplet_loss": 0.0,
        "batch_hard_loss": 0.0,
        "face_ce_loss": 0.0,
        "sketch_ce_loss": 0.0,
        "face_acc": 0.0,
        "sketch_acc": 0.0,
        "triplet_acc": 0.0,
        "grad_norm": 0.0,
    }
    steps = 0
    grad_steps = 0

    total_steps = len(data_loader)
    iterator = tqdm(data_loader, desc="Train", leave=False) if show_progress and tqdm is not None else data_loader
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(iterator, start=1):
        face_anchor = batch["face_anchor"].to(device, non_blocking=True)
        sketch_positive = batch["sketch_positive"].to(device, non_blocking=True)
        sketch_negative = batch["sketch_negative"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            face_embedding = model.embed(face_anchor)
            sketch_positive_embedding = model.embed(sketch_positive)
            sketch_negative_embedding = model.embed(sketch_negative)

            loss_triplet = triplet_loss(face_embedding, sketch_positive_embedding, sketch_negative_embedding)
            positive_similarity = F.cosine_similarity(face_embedding, sketch_positive_embedding)
            negative_similarity = F.cosine_similarity(face_embedding, sketch_negative_embedding)
            triplet_acc = (positive_similarity > negative_similarity).float().mean()
            loss = triplet_weight * loss_triplet
            loss_batch_hard = torch.zeros((), device=device)

            if batch_hard_loss is not None and batch_hard_weight > 0:
                batch_hard_embeddings, batch_hard_labels = _batch_hard_inputs(
                    batch,
                    face_embedding,
                    sketch_positive_embedding,
                    sketch_negative_embedding,
                    device,
                )
                if batch_hard_embeddings is not None:
                    loss_batch_hard = batch_hard_loss(batch_hard_embeddings, batch_hard_labels)
                    loss = loss + batch_hard_weight * loss_batch_hard

            face_ce_loss = torch.zeros((), device=device)
            sketch_ce_loss = torch.zeros((), device=device)
            face_acc = torch.zeros((), device=device)
            sketch_acc = torch.zeros((), device=device)

            if face_ce_weight > 0:
                face_label_key = "face_identity_label" if "face_identity_label" in batch else "face_binary_label"
                face_labels = batch[face_label_key].to(device, non_blocking=True)
                face_logits = model.face_head(face_embedding, face_labels)
                face_ce_loss = F.cross_entropy(face_logits, face_labels)
                face_acc = classification_accuracy(face_logits, face_labels)
                loss = loss + face_ce_weight * face_ce_loss

            if sketch_ce_weight > 0:
                sketch_label_key = "sketch_identity_label" if "sketch_identity_label" in batch else "sketch_binary_label"
                sketch_negative_label_key = (
                    "sketch_negative_identity_label"
                    if "sketch_negative_identity_label" in batch
                    else "sketch_negative_label"
                )
                sketch_labels = batch[sketch_label_key].to(device, non_blocking=True)
                sketch_embeddings = sketch_positive_embedding

                if sketch_negative_label_key in batch:
                    sketch_negative_labels = batch[sketch_negative_label_key].to(device, non_blocking=True)
                    sketch_embeddings = torch.cat([sketch_positive_embedding, sketch_negative_embedding], dim=0)
                    sketch_labels = torch.cat([sketch_labels, sketch_negative_labels], dim=0)

                sketch_logits = model.sketch_head(sketch_embeddings, sketch_labels)
                sketch_ce_loss = F.cross_entropy(sketch_logits, sketch_labels)
                sketch_acc = classification_accuracy(sketch_logits, sketch_labels)
                loss = loss + sketch_ce_weight * sketch_ce_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss detected. Try --no-amp, a lower arcface_scale, "
                    "or inspect the current batch for corrupt images/labels."
                )

        scaled_loss = loss / accumulation_steps
        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % accumulation_steps == 0 or step == total_steps:
            if track_grad_norm:
                if scaler is not None and use_amp:
                    scaler.unscale_(optimizer)
                totals["grad_norm"] += _grad_norm(model.parameters())
                grad_steps += 1
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        steps += 1
        totals["loss"] += loss.item()
        totals["triplet_loss"] += loss_triplet.item()
        totals["batch_hard_loss"] += loss_batch_hard.item()
        totals["face_ce_loss"] += face_ce_loss.item()
        totals["sketch_ce_loss"] += sketch_ce_loss.item()
        totals["face_acc"] += face_acc.item()
        totals["sketch_acc"] += sketch_acc.item()
        totals["triplet_acc"] += triplet_acc.item()

        if show_progress and log_interval and (step % log_interval == 0 or step == total_steps):
            running_loss = totals["loss"] / steps
            running_triplet_acc = totals["triplet_acc"] / steps
            print(
                f"Train step {step}/{total_steps} "
                f"loss={running_loss:.4f} triplet_acc={running_triplet_acc:.4f}",
                flush=True,
            )

    if steps == 0:
        raise ValueError("The training dataloader produced no batches.")

    averaged = {key: value / steps for key, value in totals.items() if key != "grad_norm"}
    averaged["grad_norm"] = totals["grad_norm"] / grad_steps if grad_steps else 0.0
    averaged["nonfinite_batches"] = 0
    return TrainStats(**averaged)


@torch.no_grad()
def evaluate_triplet_epoch(
    model,
    data_loader,
    device,
    triplet_loss: TripletLoss,
    batch_hard_loss: BatchHardTripletLoss | None = None,
    triplet_weight=1.0,
    batch_hard_weight=0.0,
    face_ce_weight=1.0,
    sketch_ce_weight=1.0,
    show_progress=True,
    log_interval=100,
    use_amp=False,
):
    model.eval()
    totals = {
        "loss": 0.0,
        "triplet_loss": 0.0,
        "batch_hard_loss": 0.0,
        "face_ce_loss": 0.0,
        "sketch_ce_loss": 0.0,
        "face_acc": 0.0,
        "sketch_acc": 0.0,
        "triplet_acc": 0.0,
        "grad_norm": 0.0,
    }
    steps = 0

    total_steps = len(data_loader)
    iterator = tqdm(data_loader, desc="Val", leave=False) if show_progress and tqdm is not None else data_loader
    for step, batch in enumerate(iterator, start=1):
        face_anchor = batch["face_anchor"].to(device, non_blocking=True)
        sketch_positive = batch["sketch_positive"].to(device, non_blocking=True)
        sketch_negative = batch["sketch_negative"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            face_embedding = model.embed(face_anchor)
            sketch_positive_embedding = model.embed(sketch_positive)
            sketch_negative_embedding = model.embed(sketch_negative)

            loss_triplet = triplet_loss(face_embedding, sketch_positive_embedding, sketch_negative_embedding)
            positive_similarity = F.cosine_similarity(face_embedding, sketch_positive_embedding)
            negative_similarity = F.cosine_similarity(face_embedding, sketch_negative_embedding)
            triplet_acc = (positive_similarity > negative_similarity).float().mean()
            loss = triplet_weight * loss_triplet
            loss_batch_hard = torch.zeros((), device=device)

            if batch_hard_loss is not None and batch_hard_weight > 0:
                batch_hard_embeddings, batch_hard_labels = _batch_hard_inputs(
                    batch,
                    face_embedding,
                    sketch_positive_embedding,
                    sketch_negative_embedding,
                    device,
                )
                if batch_hard_embeddings is not None:
                    loss_batch_hard = batch_hard_loss(batch_hard_embeddings, batch_hard_labels)
                    loss = loss + batch_hard_weight * loss_batch_hard

            face_ce_loss = torch.zeros((), device=device)
            sketch_ce_loss = torch.zeros((), device=device)
            face_acc = torch.zeros((), device=device)
            sketch_acc = torch.zeros((), device=device)

            if face_ce_weight > 0:
                face_label_key = "face_identity_label" if "face_identity_label" in batch else "face_binary_label"
                face_labels = batch[face_label_key].to(device, non_blocking=True)
                if int(face_labels.max().item()) < model.face_head.out_features:
                    face_logits = model.face_head(face_embedding, face_labels)
                    face_ce_loss = F.cross_entropy(face_logits, face_labels)
                    face_acc = classification_accuracy(face_logits, face_labels)
                    loss = loss + face_ce_weight * face_ce_loss

            if sketch_ce_weight > 0:
                sketch_label_key = "sketch_identity_label" if "sketch_identity_label" in batch else "sketch_binary_label"
                sketch_negative_label_key = (
                    "sketch_negative_identity_label"
                    if "sketch_negative_identity_label" in batch
                    else "sketch_negative_label"
                )
                sketch_labels = batch[sketch_label_key].to(device, non_blocking=True)
                sketch_embeddings = sketch_positive_embedding

                if sketch_negative_label_key in batch:
                    sketch_negative_labels = batch[sketch_negative_label_key].to(device, non_blocking=True)
                    sketch_embeddings = torch.cat([sketch_positive_embedding, sketch_negative_embedding], dim=0)
                    sketch_labels = torch.cat([sketch_labels, sketch_negative_labels], dim=0)

                if int(sketch_labels.max().item()) < model.sketch_head.out_features:
                    sketch_logits = model.sketch_head(sketch_embeddings, sketch_labels)
                    sketch_ce_loss = F.cross_entropy(sketch_logits, sketch_labels)
                    sketch_acc = classification_accuracy(sketch_logits, sketch_labels)
                    loss = loss + sketch_ce_weight * sketch_ce_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite validation loss detected. Try --no-amp, a lower arcface_scale, "
                    "or inspect the validation batch for corrupt images/labels."
                )

        steps += 1
        totals["loss"] += loss.item()
        totals["triplet_loss"] += loss_triplet.item()
        totals["batch_hard_loss"] += loss_batch_hard.item()
        totals["face_ce_loss"] += face_ce_loss.item()
        totals["sketch_ce_loss"] += sketch_ce_loss.item()
        totals["face_acc"] += face_acc.item()
        totals["sketch_acc"] += sketch_acc.item()
        totals["triplet_acc"] += triplet_acc.item()

        if show_progress and log_interval and (step % log_interval == 0 or step == total_steps):
            running_loss = totals["loss"] / steps
            running_triplet_acc = totals["triplet_acc"] / steps
            print(
                f"Val step {step}/{total_steps} "
                f"loss={running_loss:.4f} triplet_acc={running_triplet_acc:.4f}",
                flush=True,
            )

    if steps == 0:
        raise ValueError("The validation dataloader produced no batches.")

    averaged = {key: value / steps for key, value in totals.items()}
    averaged["nonfinite_batches"] = 0
    return TrainStats(**averaged)
