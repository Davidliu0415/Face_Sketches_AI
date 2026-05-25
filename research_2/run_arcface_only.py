from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import datasets_config as data_config
from configs import params
from data.dataset import (
    FaceSketchPairDataset,
    FaceSketchTripletDataset,
    build_identity_label_map,
    load_pair_dataset_from_test_set,
)
from eval.metrics import cosine_scores, retrieval_diagnostics, retrieval_metrics
from network import BatchHardTripletLoss, TripletLoss
from training.experiment import diagnostic_rows, git_revision, load_checkpoint_compatible, params_snapshot, write_json
from training.main import build_model, set_seed
from training.retrieval import write_diagnostics_csv
from training.train import autocast_context, evaluate_triplet_epoch, train_one_epoch

ARCFACE_ONLY_TRIPLET_WEIGHT = 0.0
ARCFACE_ONLY_BATCH_HARD_WEIGHT = 0.0
ARCFACE_ONLY_FACE_CE_WEIGHT = 1.0
ARCFACE_ONLY_SKETCH_CE_WEIGHT = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run stage-2 Face-Sketch ArcFace-only experiment.")
    parser.add_argument("--run-id", default="arcface_only_001")
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
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "research_2"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--val-retrieval-limit", type=int, default=params.val_retrieval_limit)
    parser.add_argument("--skip-val-retrieval", action="store_true")
    parser.add_argument("--seed", type=int, default=params.seed)
    parser.add_argument("--arcface-scale", type=float, default=params.arcface_scale)
    parser.add_argument("--triplet-weight", type=float, default=ARCFACE_ONLY_TRIPLET_WEIGHT)
    parser.add_argument("--batch-hard-weight", type=float, default=ARCFACE_ONLY_BATCH_HARD_WEIGHT)
    parser.add_argument("--face-ce-weight", type=float, default=ARCFACE_ONLY_FACE_CE_WEIGHT)
    parser.add_argument("--sketch-ce-weight", type=float, default=ARCFACE_ONLY_SKETCH_CE_WEIGHT)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=params.use_amp)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def apply_cli_overrides(args) -> None:
    global ARCFACE_ONLY_TRIPLET_WEIGHT
    global ARCFACE_ONLY_BATCH_HARD_WEIGHT
    global ARCFACE_ONLY_FACE_CE_WEIGHT
    global ARCFACE_ONLY_SKETCH_CE_WEIGHT

    ARCFACE_ONLY_TRIPLET_WEIGHT = args.triplet_weight
    ARCFACE_ONLY_BATCH_HARD_WEIGHT = args.batch_hard_weight
    ARCFACE_ONLY_FACE_CE_WEIGHT = args.face_ce_weight
    ARCFACE_ONLY_SKETCH_CE_WEIGHT = args.sketch_ce_weight
    params.arcface_scale = args.arcface_scale
    params.triplet_weight = args.triplet_weight
    params.batch_hard_weight = args.batch_hard_weight
    params.face_ce_weight = args.face_ce_weight
    params.sketch_ce_weight = args.sketch_ce_weight


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
            "arcface_only_weights": {
                "triplet_weight": ARCFACE_ONLY_TRIPLET_WEIGHT,
                "batch_hard_weight": ARCFACE_ONLY_BATCH_HARD_WEIGHT,
                "face_ce_weight": ARCFACE_ONLY_FACE_CE_WEIGHT,
                "sketch_ce_weight": ARCFACE_ONLY_SKETCH_CE_WEIGHT,
            },
            "config": {
                "embedding_size": params.embedding_size,
                "dropout": params.dropout,
                "face_num_classes": params.face_num_classes,
                "sketch_num_classes": params.sketch_num_classes,
                "arcface_scale": params.arcface_scale,
                "arcface_margin": params.arcface_margin,
                "triplet_margin": params.triplet_margin,
                "batch_hard_weight": ARCFACE_ONLY_BATCH_HARD_WEIGHT,
                "image_size": params.image_size,
            },
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str | Path, model, device) -> None:
    report = load_checkpoint_compatible(path, model, device)
    if report["skipped_shape_mismatch"]:
        print(f"Skipped incompatible checkpoint tensors: {len(report['skipped_shape_mismatch'])}", flush=True)


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


def retrieval_report(face_embeddings, sketch_embeddings, face_paths, sketch_paths, identities, run_id, dataset_name):
    scores = cosine_scores(face_embeddings, sketch_embeddings)
    metrics = retrieval_metrics(face_embeddings, sketch_embeddings)
    diagnostics = retrieval_diagnostics(face_embeddings, sketch_embeddings, sketch_paths=sketch_paths)
    target = torch.arange(scores.size(0), device=scores.device)
    top5_k = min(5, scores.size(1))
    _, top5_predictions = scores.topk(top5_k, dim=1)
    top1_predictions = top5_predictions[:, 0]

    top1 = top1_predictions.eq(target).float().mean().item()
    top5 = top5_predictions.eq(target.view(-1, 1)).any(dim=1).float().mean().item()
    positive_scores = scores.diag()

    failures = []
    for index, predicted_index in enumerate(top1_predictions.tolist()):
        if predicted_index == index:
            continue

        correct_score = positive_scores[index].item()
        predicted_score = scores[index, predicted_index].item()
        correct_rank = int((scores[index] > positive_scores[index]).sum().item() + 1)
        failures.append(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "identity": identities[index],
                "face_image": face_paths[index],
                "predicted_sketch": sketch_paths[predicted_index],
                "correct_sketch": sketch_paths[index],
                "similarity": f"{predicted_score:.6f}",
                "correct_similarity": f"{correct_score:.6f}",
                "rank": correct_rank,
                "error_type": "",
                "notes": "",
            }
        )

    metrics["top1"] = top1
    metrics["top5"] = top5
    metrics["mean_positive_cosine"] = positive_scores.mean().item()
    return metrics, failures, diagnostics


def run_test_set(model, test_set: dict, args, device):
    name = str(test_set["name"])
    started_at = time.perf_counter()
    dataset = load_pair_dataset_from_test_set(
        test_set,
        image_size=params.image_size,
    )
    dataset = maybe_limit(dataset, args.limit_test)
    loader = make_loader(dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    face_embeddings, sketch_embeddings, face_paths, sketch_paths, identities, _ = embed_pair_dataset(
        model,
        loader,
        device,
        use_amp=args.amp,
    )
    metrics, failures, diagnostics = retrieval_report(
        face_embeddings=face_embeddings,
        sketch_embeddings=sketch_embeddings,
        face_paths=face_paths,
        sketch_paths=sketch_paths,
        identities=identities,
        run_id=args.run_id,
        dataset_name=name,
    )
    runtime_seconds = time.perf_counter() - started_at
    print(
        f"[Test:{name}] samples={len(dataset)} "
        f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
        f"mrr={metrics['mrr']:.4f} map={metrics['map']:.4f} "
        f"positive_cos={metrics['mean_positive_cosine']:.4f} "
        f"runtime={runtime_seconds:.2f}s",
        flush=True,
    )
    return {"name": name, "samples": len(dataset), "runtime_seconds": runtime_seconds, **metrics}, failures, diagnostics


def write_arcface_only_results(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_id",
        "dataset",
        "train_samples",
        "val_samples",
        "test_samples",
        "top1",
        "top5",
        "mrr",
        "map",
        "mean_positive_cosine",
        "runtime_seconds",
        "checkpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failed_matches(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_id",
        "dataset",
        "identity",
        "face_image",
        "predicted_sketch",
        "correct_sketch",
        "similarity",
        "correct_similarity",
        "rank",
        "error_type",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    apply_cli_overrides(args)
    started_at = time.perf_counter()
    set_seed(args.seed)
    device = params.device

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.run_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    metrics_path = run_dir / "metrics.csv"
    results_path = run_dir / "arcface_only_results.csv"
    failures_path = run_dir / "failed_matches.csv"
    diagnostics_path = run_dir / "retrieval_diagnostics.csv"
    config_path = run_dir / "config.json"

    identity_to_label = build_identity_label_map(args.train_manifest)
    model = build_model(device, face_num_classes=len(identity_to_label), sketch_num_classes=len(identity_to_label))
    scaler = build_grad_scaler(device, args.amp)
    criterion = TripletLoss(margin=params.triplet_margin)
    batch_hard = BatchHardTripletLoss(margin=params.triplet_margin)
    write_json(
        config_path,
        {
            "args": vars(args),
            "git": git_revision(PROJECT_ROOT),
            "params": params_snapshot(params, num_identities=len(identity_to_label)),
            "stage_weights": {
                "triplet_weight": ARCFACE_ONLY_TRIPLET_WEIGHT,
                "batch_hard_weight": ARCFACE_ONLY_BATCH_HARD_WEIGHT,
                "face_ce_weight": ARCFACE_ONLY_FACE_CE_WEIGHT,
                "sketch_ce_weight": ARCFACE_ONLY_SKETCH_CE_WEIGHT,
            },
        },
    )

    print(f"Run ID: {args.run_id}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Output dir: {run_dir}", flush=True)
    print(
        "ArcFace-only weights: "
        f"triplet={ARCFACE_ONLY_TRIPLET_WEIGHT}, "
        f"batch_hard={ARCFACE_ONLY_BATCH_HARD_WEIGHT}, "
        f"face_ce={ARCFACE_ONLY_FACE_CE_WEIGHT}, "
        f"sketch_ce={ARCFACE_ONLY_SKETCH_CE_WEIGHT}",
        flush=True,
    )

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device)
        print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    train_samples = ""
    val_samples = ""
    checkpoint_for_results = ""
    best_val_loss = float("inf")
    best_selection_value = float("-inf")

    if not args.skip_train:
        train_dataset = FaceSketchTripletDataset(
            args.train_manifest,
            args.image_root,
            image_size=params.image_size,
            identity_to_label=identity_to_label,
        )
        train_dataset = maybe_limit(train_dataset, args.limit_train)
        train_samples = len(train_dataset)
        train_loader = make_loader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=len(train_dataset) >= args.batch_size,
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
            val_samples = len(val_dataset)
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
                    "train_face_ce_loss",
                    "train_sketch_ce_loss",
                    "train_triplet_acc",
                    "train_face_acc",
                    "train_sketch_acc",
                    "train_grad_norm",
                    "val_loss",
                    "val_triplet_loss",
                    "val_batch_hard_loss",
                    "val_face_ce_loss",
                    "val_sketch_ce_loss",
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
                    triplet_weight=ARCFACE_ONLY_TRIPLET_WEIGHT,
                    batch_hard_weight=ARCFACE_ONLY_BATCH_HARD_WEIGHT,
                    face_ce_weight=ARCFACE_ONLY_FACE_CE_WEIGHT,
                    sketch_ce_weight=ARCFACE_ONLY_SKETCH_CE_WEIGHT,
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
                        triplet_weight=ARCFACE_ONLY_TRIPLET_WEIGHT,
                        batch_hard_weight=ARCFACE_ONLY_BATCH_HARD_WEIGHT,
                        face_ce_weight=ARCFACE_ONLY_FACE_CE_WEIGHT if val_labels_known else 0.0,
                        sketch_ce_weight=ARCFACE_ONLY_SKETCH_CE_WEIGHT if val_labels_known else 0.0,
                        log_interval=args.log_interval,
                        use_amp=args.amp,
                    )
                    if val_retrieval_loader is not None:
                        val_face, val_sketch, _, _, _, val_labels = embed_pair_dataset(
                            model,
                            val_retrieval_loader,
                            device,
                            use_amp=args.amp,
                        )
                        val_retrieval_metrics = retrieval_metrics(
                            val_face,
                            val_sketch,
                            face_labels=val_labels,
                            sketch_labels=val_labels,
                        )
                    selection_value = val_retrieval_metrics.get("mrr", -val_stats.loss)
                    if selection_value > best_selection_value:
                        best_selection_value = selection_value
                        best_val_loss = val_stats.loss
                        save_checkpoint(best_path, model, epoch, best_val_loss, args)

                save_checkpoint(last_path, model, epoch, val_stats.loss if val_stats else None, args)
                if val_stats is None and not best_path.exists():
                    save_checkpoint(best_path, model, epoch, None, args)

                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": train_stats.loss,
                        "train_triplet_loss": train_stats.triplet_loss,
                        "train_batch_hard_loss": train_stats.batch_hard_loss,
                        "train_face_ce_loss": train_stats.face_ce_loss,
                        "train_sketch_ce_loss": train_stats.sketch_ce_loss,
                        "train_triplet_acc": train_stats.triplet_acc,
                        "train_face_acc": train_stats.face_acc,
                        "train_sketch_acc": train_stats.sketch_acc,
                        "train_grad_norm": train_stats.grad_norm,
                        "val_loss": val_stats.loss if val_stats else "",
                        "val_triplet_loss": val_stats.triplet_loss if val_stats else "",
                        "val_batch_hard_loss": val_stats.batch_hard_loss if val_stats else "",
                        "val_face_ce_loss": val_stats.face_ce_loss if val_stats else "",
                        "val_sketch_ce_loss": val_stats.sketch_ce_loss if val_stats else "",
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
                        f"train_loss={train_stats.loss:.4f} "
                        f"train_face_acc={train_stats.face_acc:.4f} "
                        f"train_sketch_acc={train_stats.sketch_acc:.4f} "
                        f"val_loss={val_stats.loss:.4f} "
                        f"val_face_acc={val_stats.face_acc:.4f} "
                        f"val_sketch_acc={val_stats.sketch_acc:.4f} "
                        f"val_mrr={val_retrieval_metrics.get('mrr', 0.0):.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Epoch {epoch}/{args.epochs}] "
                        f"train_loss={train_stats.loss:.4f} "
                        f"train_face_acc={train_stats.face_acc:.4f} "
                        f"train_sketch_acc={train_stats.sketch_acc:.4f}",
                        flush=True,
                    )

        if best_path.exists():
            load_checkpoint(best_path, model, device)
            checkpoint_for_results = str(best_path)
            print(f"Testing with best checkpoint: {best_path}", flush=True)
        elif last_path.exists():
            checkpoint_for_results = str(last_path)
            print(f"Testing with last checkpoint: {last_path}", flush=True)

    elif not args.skip_val:
        val_dataset = FaceSketchTripletDataset(
            args.val_manifest,
            args.image_root,
            image_size=params.image_size,
            identity_to_label=build_identity_label_map(args.val_manifest),
        )
        val_dataset = maybe_limit(val_dataset, args.limit_val)
        val_samples = len(val_dataset)
        val_loader = make_loader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        val_stats = evaluate_triplet_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            triplet_loss=criterion,
            batch_hard_loss=batch_hard,
            triplet_weight=ARCFACE_ONLY_TRIPLET_WEIGHT,
            batch_hard_weight=ARCFACE_ONLY_BATCH_HARD_WEIGHT,
            face_ce_weight=0.0,
            sketch_ce_weight=0.0,
            log_interval=args.log_interval,
            use_amp=args.amp,
        )
        print(
            f"[Val] loss={val_stats.loss:.4f} "
            f"face_acc={val_stats.face_acc:.4f} sketch_acc={val_stats.sketch_acc:.4f}",
            flush=True,
        )

    if args.checkpoint:
        checkpoint_for_results = args.checkpoint

    result_rows = []
    failed_match_rows = []
    diagnostics_rows = []
    if not args.skip_test:
        for test_set in data_config.get_test_sets():
            name = str(test_set["name"])
            test_result, failures, diagnostics = run_test_set(model, test_set, args, device)
            result_rows.append(
                {
                    "run_id": args.run_id,
                    "dataset": name,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "test_samples": test_result["samples"],
                    "top1": f"{test_result['top1']:.6f}",
                    "top5": f"{test_result['top5']:.6f}",
                    "mrr": f"{test_result['mrr']:.6f}",
                    "map": f"{test_result['map']:.6f}",
                    "mean_positive_cosine": f"{test_result['mean_positive_cosine']:.6f}",
                    "runtime_seconds": f"{test_result['runtime_seconds']:.2f}",
                    "checkpoint": checkpoint_for_results,
                }
            )
            failed_match_rows.extend(failures)
            diagnostics_rows.extend(diagnostic_rows(args.run_id, name, diagnostics))

    write_arcface_only_results(results_path, result_rows)
    write_failed_matches(failures_path, failed_match_rows)
    write_diagnostics_csv(diagnostics_path, diagnostics_rows)

    total_runtime = time.perf_counter() - started_at
    print(f"arcface_only_results.csv: {results_path}", flush=True)
    print(f"failed_matches.csv: {failures_path}", flush=True)
    print(f"Total runtime: {total_runtime:.2f}s", flush=True)


if __name__ == "__main__":
    main()
