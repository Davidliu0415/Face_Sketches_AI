from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import datasets_config as data_config
from configs import params
from data.dataset import (
    FaceSketchPairDataset,
    FaceSketchTripletDataset,
    build_identity_label_map,
    load_pair_dataset_from_test_set,
)
from network import BatchHardTripletLoss, TripletLoss
from training.experiment import diagnostic_rows, git_revision, load_checkpoint_compatible, params_snapshot, write_json
from training.main import build_model, run_dry_check, set_seed
from training.retrieval import retrieval_report, write_diagnostics_csv
from training.train import evaluate_triplet_epoch, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, and test the face-sketch matcher.")
    parser.add_argument("--train-manifest", default=str(data_config.train_manifest))
    parser.add_argument("--val-manifest", default=str(data_config.val_manifest))
    parser.add_argument("--image-root", default=str(data_config.image_root))
    parser.add_argument("--epochs", type=int, default=params.epochs)
    parser.add_argument("--batch-size", type=int, default=params.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=params.eval_batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=params.accumulation_steps)
    parser.add_argument("--lr", type=float, default=params.lr)
    parser.add_argument("--num-workers", type=int, default=params.num_workers)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--val-retrieval-limit", type=int, default=params.val_retrieval_limit)
    parser.add_argument("--skip-val-retrieval", action="store_true")
    parser.add_argument("--seed", type=int, default=params.seed)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=params.use_amp)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def build_grad_scaler(device, enabled):
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def maybe_limit(dataset, limit: int):
    if limit and limit > 0:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


def save_checkpoint(path: Path, model, epoch: int, val_loss: float | None, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": {
                "embedding_size": params.embedding_size,
                "dropout": params.dropout,
                "face_num_classes": params.face_num_classes,
                "sketch_num_classes": params.sketch_num_classes,
                "arcface_scale": params.arcface_scale,
                "arcface_margin": params.arcface_margin,
                "triplet_margin": params.triplet_margin,
                "batch_hard_weight": params.batch_hard_weight,
            },
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str | Path, model, device) -> None:
    report = load_checkpoint_compatible(path, model, device)
    if report["skipped_shape_mismatch"]:
        print(f"Skipped incompatible checkpoint tensors: {len(report['skipped_shape_mismatch'])}", flush=True)


def run_test_set(model, test_set: dict, args, device):
    name = str(test_set["name"])
    dataset = load_pair_dataset_from_test_set(
        test_set,
        image_size=params.image_size,
    )
    dataset = maybe_limit(dataset, args.limit_test)
    loader = make_loader(dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    metrics, diagnostics, failures = retrieval_report(
        model,
        loader,
        device,
        use_amp=args.amp,
        run_id="train_val_test",
        dataset_name=name,
    )
    print(
        f"[Test:{name}] samples={len(dataset)} "
        f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
        f"mrr={metrics['mrr']:.4f} map={metrics['map']:.4f} "
        f"positive_cos={metrics['mean_positive_cosine']:.4f}",
        flush=True,
    )
    return {"name": name, "samples": len(dataset), **metrics}, diagnostics, failures


def main():
    args = parse_args()
    set_seed(args.seed)
    device = params.device

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    print(f"Device: {device}", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(
        f"4090D-24GB profile: batch_size={args.batch_size}, "
        f"eval_batch_size={args.eval_batch_size}, accumulation_steps={args.accumulation_steps}, amp={args.amp}",
        flush=True,
    )

    if args.dry_run:
        model = build_model(device)
        run_dry_check(model, device)
        return

    identity_to_label = build_identity_label_map(args.train_manifest)
    model = build_model(device, face_num_classes=len(identity_to_label), sketch_num_classes=len(identity_to_label))
    scaler = build_grad_scaler(device, args.amp)
    criterion = TripletLoss(margin=params.triplet_margin)
    batch_hard = BatchHardTripletLoss(margin=params.triplet_margin)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device)
        print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"train_val_test_{run_id}"
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    test_results_path = run_dir / "test_results.csv"
    failures_path = run_dir / "failed_matches.csv"
    diagnostics_path = run_dir / "retrieval_diagnostics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_mrr = float("-inf")
    write_json(
        config_path,
        {
            "args": vars(args),
            "git": git_revision(PROJECT_ROOT),
            "params": params_snapshot(params, num_identities=len(identity_to_label)),
            "identity_count": len(identity_to_label),
        },
    )

    if not args.skip_train:
        train_dataset = FaceSketchTripletDataset(
            args.train_manifest,
            args.image_root,
            image_size=params.image_size,
            identity_to_label=identity_to_label,
        )
        train_dataset = maybe_limit(train_dataset, args.limit_train)
        train_loader = make_loader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=params.weight_decay)

        val_loader = None
        val_retrieval_loader = None
        if not args.skip_val:
            val_identity_to_label = build_identity_label_map(args.val_manifest)
            val_labels_known = set(val_identity_to_label).issubset(identity_to_label)
            val_dataset_identity_to_label = identity_to_label if val_labels_known else val_identity_to_label
            val_dataset = FaceSketchTripletDataset(
                args.val_manifest,
                args.image_root,
                image_size=params.image_size,
                identity_to_label=val_dataset_identity_to_label,
            )
            val_dataset = maybe_limit(val_dataset, args.limit_val)
            val_loader = make_loader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            if not args.skip_val_retrieval:
                val_retrieval_dataset = FaceSketchPairDataset.from_manifest(
                    args.val_manifest,
                    args.image_root,
                    image_size=params.image_size,
                    identity_to_label=val_dataset_identity_to_label,
                )
                val_retrieval_dataset = maybe_limit(val_retrieval_dataset, args.val_retrieval_limit)
                val_retrieval_loader = make_loader(
                    val_retrieval_dataset,
                    args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )

        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_triplet_loss",
                    "train_batch_hard_loss",
                    "train_triplet_acc",
                    "train_face_acc",
                    "train_sketch_acc",
                    "train_grad_norm",
                    "val_loss",
                    "val_triplet_loss",
                    "val_batch_hard_loss",
                    "val_triplet_acc",
                    "val_face_acc",
                    "val_sketch_acc",
                    "val_retrieval_top1",
                    "val_retrieval_top5",
                    "val_retrieval_mrr",
                    "val_retrieval_map",
                    "val_retrieval_mean_positive_cosine",
                ],
            )
            writer.writeheader()

            for epoch in range(1, args.epochs + 1):
                train_stats = train_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    triplet_loss=criterion,
                    batch_hard_loss=batch_hard,
                    triplet_weight=params.triplet_weight,
                    batch_hard_weight=params.batch_hard_weight,
                    face_ce_weight=params.face_ce_weight,
                    sketch_ce_weight=params.sketch_ce_weight,
                    log_interval=args.log_interval,
                    accumulation_steps=args.accumulation_steps,
                    use_amp=args.amp,
                    scaler=scaler,
                    track_grad_norm=True,
                )

                val_stats = None
                val_retrieval_metrics = {}
                if val_loader is not None:
                    val_stats = evaluate_triplet_epoch(
                        model=model,
                        data_loader=val_loader,
                        device=device,
                        triplet_loss=criterion,
                        batch_hard_loss=batch_hard,
                        triplet_weight=params.triplet_weight,
                        batch_hard_weight=params.batch_hard_weight,
                        face_ce_weight=params.face_ce_weight if val_labels_known else 0.0,
                        sketch_ce_weight=params.sketch_ce_weight if val_labels_known else 0.0,
                        log_interval=args.log_interval,
                        use_amp=args.amp,
                    )
                    if val_retrieval_loader is not None:
                        val_retrieval_metrics, _, _ = retrieval_report(
                            model,
                            val_retrieval_loader,
                            device,
                            use_amp=args.amp,
                            run_id="train_val_test",
                            dataset_name="val",
                            include_failures=False,
                        )
                    if val_retrieval_metrics and val_retrieval_metrics["mrr"] > best_val_mrr:
                        best_val_mrr = val_retrieval_metrics["mrr"]
                        best_val_loss = val_stats.loss
                        save_checkpoint(best_path, model, epoch, best_val_loss, args)
                    elif not val_retrieval_metrics and val_stats.loss < best_val_loss:
                        best_val_loss = val_stats.loss
                        save_checkpoint(best_path, model, epoch, best_val_loss, args)

                save_checkpoint(last_path, model, epoch, val_stats.loss if val_stats else None, args)
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": train_stats.loss,
                        "train_triplet_loss": train_stats.triplet_loss,
                        "train_batch_hard_loss": train_stats.batch_hard_loss,
                        "train_triplet_acc": train_stats.triplet_acc,
                        "train_face_acc": train_stats.face_acc,
                        "train_sketch_acc": train_stats.sketch_acc,
                        "train_grad_norm": train_stats.grad_norm,
                        "val_loss": val_stats.loss if val_stats else "",
                        "val_triplet_loss": val_stats.triplet_loss if val_stats else "",
                        "val_batch_hard_loss": val_stats.batch_hard_loss if val_stats else "",
                        "val_triplet_acc": val_stats.triplet_acc if val_stats else "",
                        "val_face_acc": val_stats.face_acc if val_stats else "",
                        "val_sketch_acc": val_stats.sketch_acc if val_stats else "",
                        "val_retrieval_top1": val_retrieval_metrics.get("top1", ""),
                        "val_retrieval_top5": val_retrieval_metrics.get("top5", ""),
                        "val_retrieval_mrr": val_retrieval_metrics.get("mrr", ""),
                        "val_retrieval_map": val_retrieval_metrics.get("map", ""),
                        "val_retrieval_mean_positive_cosine": val_retrieval_metrics.get("mean_positive_cosine", ""),
                    }
                )
                handle.flush()

                if val_stats:
                    print(
                        f"[Epoch {epoch}/{args.epochs}] "
                        f"train_loss={train_stats.loss:.4f} train_triplet_acc={train_stats.triplet_acc:.4f} "
                        f"val_loss={val_stats.loss:.4f} val_triplet_acc={val_stats.triplet_acc:.4f} "
                        f"val_mrr={val_retrieval_metrics.get('mrr', 0.0):.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Epoch {epoch}/{args.epochs}] "
                        f"train_loss={train_stats.loss:.4f} train_triplet_acc={train_stats.triplet_acc:.4f}",
                        flush=True,
                    )

        if best_path.exists():
            load_checkpoint(best_path, model, device)
            print(f"Testing with best checkpoint: {best_path}", flush=True)
        else:
            print(f"Testing with last checkpoint: {last_path}", flush=True)

    elif not args.skip_val:
        val_dataset = FaceSketchTripletDataset(
            args.val_manifest,
            args.image_root,
            image_size=params.image_size,
            identity_to_label=build_identity_label_map(args.val_manifest),
        )
        val_dataset = maybe_limit(val_dataset, args.limit_val)
        val_loader = make_loader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        val_stats = evaluate_triplet_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            triplet_loss=criterion,
            batch_hard_loss=batch_hard,
            triplet_weight=params.triplet_weight,
            batch_hard_weight=params.batch_hard_weight,
            face_ce_weight=0.0,
            sketch_ce_weight=0.0,
            log_interval=args.log_interval,
            use_amp=args.amp,
        )
        print(f"[Val] loss={val_stats.loss:.4f} triplet_acc={val_stats.triplet_acc:.4f}", flush=True)

    if not args.skip_test:
        test_rows = []
        failure_rows = []
        diagnostics_rows = []
        for test_set in data_config.get_test_sets():
            name = str(test_set["name"])
            test_result, diagnostics, failures = run_test_set(model, test_set, args, device)
            test_rows.append(
                {
                    "dataset": name,
                    "test_samples": test_result["samples"],
                    "top1": f"{test_result['top1']:.6f}",
                    "top5": f"{test_result['top5']:.6f}",
                    "mrr": f"{test_result['mrr']:.6f}",
                    "map": f"{test_result['map']:.6f}",
                    "mean_positive_cosine": f"{test_result['mean_positive_cosine']:.6f}",
                }
            )
            failure_rows.extend(failures)
            diagnostics_rows.extend(diagnostic_rows("train_val_test", name, diagnostics))

        with test_results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "test_samples", "top1", "top5", "mrr", "map", "mean_positive_cosine"],
            )
            writer.writeheader()
            writer.writerows(test_rows)
        if failure_rows:
            with failures_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0].keys()))
                writer.writeheader()
                writer.writerows(failure_rows)
        write_diagnostics_csv(diagnostics_path, diagnostics_rows)

    print(f"Output dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
