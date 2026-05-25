from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import datasets_config as data_config
from configs import params
from data.dataset import FaceSketchTripletDataset, build_identity_label_map
from network import BatchHardTripletLoss, FaceSketchMatcher, TripletLoss
from training.train import train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileFaceNet face/sketch matcher.")
    parser.add_argument("--manifest", default=str(data_config.train_manifest), help="CSV manifest path.")
    parser.add_argument("--image-root", default=str(data_config.image_root), help="Root for relative image paths.")
    parser.add_argument("--epochs", type=int, default=params.epochs)
    parser.add_argument("--batch-size", type=int, default=params.batch_size)
    parser.add_argument("--lr", type=float, default=params.lr)
    parser.add_argument("--dry-run", action="store_true", help="Run a synthetic forward/backward check.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device, face_num_classes=None, sketch_num_classes=None):
    model = FaceSketchMatcher(
        embedding_size=params.embedding_size,
        dropout=params.dropout,
        face_num_classes=face_num_classes or params.face_num_classes,
        sketch_num_classes=sketch_num_classes or params.sketch_num_classes,
        arcface_scale=params.arcface_scale,
        arcface_margin=params.arcface_margin,
    )
    return model.to(device)


def run_dry_check(model, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    criterion = TripletLoss(margin=params.triplet_margin)
    batch_hard = BatchHardTripletLoss(margin=params.triplet_margin)

    batch_size = 2
    face = torch.randn(batch_size, 3, params.image_size, params.image_size, device=device)
    sketch_pos = torch.randn(batch_size, 3, params.image_size, params.image_size, device=device)
    sketch_neg = torch.randn(batch_size, 3, params.image_size, params.image_size, device=device)
    face_labels = torch.arange(batch_size, dtype=torch.long, device=device)
    sketch_labels = torch.arange(batch_size, dtype=torch.long, device=device)
    sketch_neg_labels = torch.arange(batch_size, dtype=torch.long, device=device).roll(1)

    face_emb = model.embed(face)
    pos_emb = model.embed(sketch_pos)
    neg_emb = model.embed(sketch_neg)
    loss = criterion(face_emb, pos_emb, neg_emb)

    face_logits = model.face_head(face_emb, face_labels)
    sketch_logits = model.sketch_head(torch.cat([pos_emb, neg_emb], dim=0), torch.cat([sketch_labels, sketch_neg_labels], dim=0))
    loss = loss + torch.nn.functional.cross_entropy(face_logits, face_labels)
    loss = loss + torch.nn.functional.cross_entropy(sketch_logits, torch.cat([sketch_labels, sketch_neg_labels], dim=0))
    loss = loss + batch_hard(
        torch.cat([face_emb, pos_emb, neg_emb], dim=0),
        torch.cat([face_labels, sketch_labels, sketch_neg_labels], dim=0),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("Dry run passed.")
    print(f"face_embedding_shape={tuple(face_emb.shape)}")
    print(f"sketch_embedding_shape={tuple(pos_emb.shape)}")
    print(f"loss={loss.item():.4f}")


def main():
    args = parse_args()
    set_seed(params.seed)
    device = params.device
    print(f"Running on device: {device}")

    if args.dry_run:
        model = build_model(device)
        run_dry_check(model, device)
        return

    identity_to_label = build_identity_label_map(args.manifest)
    model = build_model(device, face_num_classes=len(identity_to_label), sketch_num_classes=len(identity_to_label))

    dataset = FaceSketchTripletDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        image_size=params.image_size,
        identity_to_label=identity_to_label,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=params.weight_decay)
    criterion = TripletLoss(margin=params.triplet_margin)
    batch_hard = BatchHardTripletLoss(margin=params.triplet_margin)

    for epoch in range(args.epochs):
        stats = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            triplet_loss=criterion,
            batch_hard_loss=batch_hard,
            triplet_weight=params.triplet_weight,
            batch_hard_weight=params.batch_hard_weight,
            face_ce_weight=params.face_ce_weight,
            sketch_ce_weight=params.sketch_ce_weight,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={stats.loss:.4f} triplet={stats.triplet_loss:.4f} "
            f"face_ce={stats.face_ce_loss:.4f} sketch_ce={stats.sketch_ce_loss:.4f}"
        )

    if params.save:
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "face_sketch_matcher.pth"
        config_snapshot = {
            "embedding_size": params.embedding_size,
            "dropout": params.dropout,
            "face_num_classes": params.face_num_classes,
            "sketch_num_classes": params.sketch_num_classes,
            "num_identities": len(identity_to_label),
            "arcface_scale": params.arcface_scale,
            "arcface_margin": params.arcface_margin,
            "triplet_margin": params.triplet_margin,
            "batch_hard_weight": params.batch_hard_weight,
        }
        torch.save({"model": model.state_dict(), "config": config_snapshot}, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
