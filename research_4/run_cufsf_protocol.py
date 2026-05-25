from __future__ import annotations

import argparse
import csv
import random
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
    image_files,
    normalized_pair_key,
)
from eval.metrics import retrieval_diagnostics, retrieval_failures, retrieval_metrics
from network import BatchHardTripletLoss, TripletLoss
from training.experiment import diagnostic_rows, git_revision, load_checkpoint_compatible, params_snapshot, write_json
from training.main import build_model, set_seed
from training.retrieval import embed_pair_dataset, write_diagnostics_csv
from training.train import evaluate_triplet_epoch, train_one_epoch

CUFSF_TRIPLET_WEIGHT = 1.0
CUFSF_BATCH_HARD_WEIGHT = 0.75
CUFSF_FACE_CE_WEIGHT = 0.0
CUFSF_SKETCH_CE_WEIGHT = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run research-4 CUFSF 500/694 identity-split protocol.")
    parser.add_argument("--run-id", default="cufsf_protocol_001")
    parser.add_argument("--face-dir", default=str(Path(data_config.test_dataset01) / "archive" / "photos"))
    parser.add_argument("--sketch-dir", default=str(Path(data_config.test_dataset01) / "archive" / "sketches"))
    parser.add_argument("--image-root", default=str(PROJECT_ROOT))
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=params.epochs)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=params.lr)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "research_4"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=params.seed)
    parser.add_argument("--arcface-scale", type=float, default=params.arcface_scale)
    parser.add_argument("--triplet-weight", type=float, default=CUFSF_TRIPLET_WEIGHT)
    parser.add_argument("--batch-hard-weight", type=float, default=CUFSF_BATCH_HARD_WEIGHT)
    parser.add_argument("--face-ce-weight", type=float, default=CUFSF_FACE_CE_WEIGHT)
    parser.add_argument("--sketch-ce-weight", type=float, default=CUFSF_SKETCH_CE_WEIGHT)
    parser.add_argument("--negative-resample-each-epoch", action="store_true")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=params.use_amp)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def apply_cli_overrides(args) -> None:
    global CUFSF_TRIPLET_WEIGHT
    global CUFSF_BATCH_HARD_WEIGHT
    global CUFSF_FACE_CE_WEIGHT
    global CUFSF_SKETCH_CE_WEIGHT

    CUFSF_TRIPLET_WEIGHT = args.triplet_weight
    CUFSF_BATCH_HARD_WEIGHT = args.batch_hard_weight
    CUFSF_FACE_CE_WEIGHT = args.face_ce_weight
    CUFSF_SKETCH_CE_WEIGHT = args.sketch_ce_weight
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


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def collect_pairs(face_dir: str | Path, sketch_dir: str | Path) -> list[tuple[Path, Path, str]]:
    face_files = image_files(face_dir)
    sketch_files = image_files(sketch_dir)
    sketch_by_key = {normalized_pair_key(path): path for path in sketch_files}

    pairs = []
    for face_path in face_files:
        identity = normalized_pair_key(face_path)
        sketch_path = sketch_by_key.get(identity)
        if sketch_path is not None:
            pairs.append((face_path, sketch_path, identity))

    if not pairs:
        raise ValueError(f"No paired CUFSF samples found in {face_dir} and {sketch_dir}.")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path, str]],
    train_count: int,
    seed: int,
) -> tuple[list[tuple[Path, Path, str]], list[tuple[Path, Path, str]]]:
    if train_count < 2:
        raise ValueError("--train-count must be at least 2 so negative identities can be sampled.")
    if train_count >= len(pairs):
        raise ValueError("--train-count must be smaller than the total paired sample count.")

    rng = random.Random(seed)
    identities = [identity for _, _, identity in pairs]
    shuffled = identities[:]
    rng.shuffle(shuffled)
    train_identities = set(shuffled[:train_count])

    train_pairs = [pair for pair in pairs if pair[2] in train_identities]
    test_pairs = [pair for pair in pairs if pair[2] not in train_identities]
    return train_pairs, test_pairs


def build_triplet_rows(
    pairs: list[tuple[Path, Path, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    rows = []
    identities = [identity for _, _, identity in pairs]
    by_identity = {identity: (face_path, sketch_path) for face_path, sketch_path, identity in pairs}

    for identity in identities:
        face_path, sketch_path = by_identity[identity]
        negative_candidates = [candidate for candidate in identities if candidate != identity]
        negative_identity = rng.choice(negative_candidates)
        _, negative_sketch_path = by_identity[negative_identity]
        rows.append(
            {
                "face_anchor": relative_path(face_path),
                "sketch_positive": relative_path(sketch_path),
                "sketch_negative": relative_path(negative_sketch_path),
                "face_binary_label": "1",
                "sketch_binary_label": "1",
                "sketch_negative_label": "0",
                "identity": identity,
                "negative_identity": negative_identity,
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_train_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(
        path,
        [
            "face_anchor",
            "sketch_positive",
            "sketch_negative",
            "face_binary_label",
            "sketch_binary_label",
            "sketch_negative_label",
            "identity",
            "negative_identity",
        ],
        rows,
    )


def write_pair_manifest(path: Path, pairs: list[tuple[Path, Path, str]]) -> None:
    rows = [
        {
            "face_anchor": relative_path(face_path),
            "sketch_positive": relative_path(sketch_path),
            "identity": identity,
        }
        for face_path, sketch_path, identity in pairs
    ]
    write_csv(path, ["face_anchor", "sketch_positive", "identity"], rows)


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    train_loss: float | None,
    args,
    train_manifest: Path,
    test_pairs: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "cufsf_weights": {
                "triplet_weight": CUFSF_TRIPLET_WEIGHT,
                "batch_hard_weight": CUFSF_BATCH_HARD_WEIGHT,
                "face_ce_weight": CUFSF_FACE_CE_WEIGHT,
                "sketch_ce_weight": CUFSF_SKETCH_CE_WEIGHT,
            },
            "config": {
                "embedding_size": params.embedding_size,
                "dropout": params.dropout,
                "arcface_scale": params.arcface_scale,
                "arcface_margin": params.arcface_margin,
                "triplet_margin": params.triplet_margin,
                "batch_hard_weight": CUFSF_BATCH_HARD_WEIGHT,
                "image_size": params.image_size,
                "train_manifest": str(train_manifest),
                "test_pairs": str(test_pairs),
            },
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str | Path, model, device) -> None:
    report = load_checkpoint_compatible(path, model, device)
    if report["skipped_shape_mismatch"]:
        print(f"Skipped incompatible checkpoint tensors: {len(report['skipped_shape_mismatch'])}", flush=True)


def write_results(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_id",
        "dataset",
        "train_identities",
        "test_identities",
        "test_samples",
        "top1",
        "top5",
        "top10",
        "top50",
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


def run_cufsf_test(model, pairs: list[tuple[Path, Path, str]], args, device):
    started_at = time.perf_counter()
    identity_to_label = {identity: index for index, (_, _, identity) in enumerate(pairs)}
    dataset = FaceSketchPairDataset(
        samples=pairs,
        image_size=params.image_size,
        identity_to_label=identity_to_label,
    )
    dataset = maybe_limit(dataset, args.limit_test)
    loader = make_loader(dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    face_embeddings, sketch_embeddings, face_paths, sketch_paths, identities, labels = embed_pair_dataset(
        model,
        loader,
        device,
        use_amp=args.amp,
    )
    metrics = retrieval_metrics(
        face_embeddings,
        sketch_embeddings,
        face_labels=labels,
        sketch_labels=labels,
        topk=(1, 5, 10, 50),
    )
    diagnostics = retrieval_diagnostics(face_embeddings, sketch_embeddings, sketch_paths=sketch_paths)
    failures = retrieval_failures(
        face_embeddings=face_embeddings,
        sketch_embeddings=sketch_embeddings,
        face_paths=face_paths,
        sketch_paths=sketch_paths,
        identities=identities,
        run_id=args.run_id,
        dataset_name="cufsf_500_694",
        face_labels=labels,
        sketch_labels=labels,
    )
    runtime_seconds = time.perf_counter() - started_at
    print(
        f"[Test:CUFSF 500/694] samples={len(dataset)} "
        f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
        f"top10={metrics['top10']:.4f} top50={metrics['top50']:.4f} "
        f"mrr={metrics['mrr']:.4f} map={metrics['map']:.4f} "
        f"positive_cos={metrics['mean_positive_cosine']:.4f} "
        f"runtime={runtime_seconds:.2f}s",
        flush=True,
    )
    return {"samples": len(dataset), "runtime_seconds": runtime_seconds, **metrics}, failures, diagnostics


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

    train_manifest_path = run_dir / "cufsf_train_manifest.csv"
    test_pairs_path = run_dir / "cufsf_test_pairs.csv"
    split_summary_path = run_dir / "split_summary.json"
    metrics_path = run_dir / "metrics.csv"
    results_path = run_dir / "cufsf_results.csv"
    failures_path = run_dir / "failed_matches.csv"
    diagnostics_path = run_dir / "retrieval_diagnostics.csv"
    config_path = run_dir / "config.json"
    best_train_path = run_dir / "best_train_loss.pth"
    last_path = run_dir / "last.pth"

    all_pairs = collect_pairs(args.face_dir, args.sketch_dir)
    train_pairs, test_pairs = split_pairs(all_pairs, args.train_count, args.seed)
    rng = random.Random(args.seed)
    train_rows = build_triplet_rows(train_pairs, rng)
    write_train_manifest(train_manifest_path, train_rows)
    write_pair_manifest(test_pairs_path, test_pairs)

    train_identities = {identity for _, _, identity in train_pairs}
    test_identities = {identity for _, _, identity in test_pairs}
    split_summary = {
        "protocol": "CUFSF 500/694 identity split",
        "seed": args.seed,
        "paired_samples": len(all_pairs),
        "train_identities": len(train_identities),
        "test_identities": len(test_identities),
        "identity_overlap": len(train_identities.intersection(test_identities)),
        "train_manifest": str(train_manifest_path),
        "test_pairs": str(test_pairs_path),
        "face_dir": str(Path(args.face_dir)),
        "sketch_dir": str(Path(args.sketch_dir)),
    }
    write_json(split_summary_path, split_summary)

    identity_to_label = build_identity_label_map(train_manifest_path)
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
            "split": split_summary,
            "stage_weights": {
                "triplet_weight": CUFSF_TRIPLET_WEIGHT,
                "batch_hard_weight": CUFSF_BATCH_HARD_WEIGHT,
                "face_ce_weight": CUFSF_FACE_CE_WEIGHT,
                "sketch_ce_weight": CUFSF_SKETCH_CE_WEIGHT,
            },
        },
    )

    print(f"Run ID: {args.run_id}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Output dir: {run_dir}", flush=True)
    print(
        "CUFSF split: "
        f"paired={len(all_pairs)}, train={len(train_pairs)}, test={len(test_pairs)}, "
        f"overlap={split_summary['identity_overlap']}",
        flush=True,
    )
    print(
        "Weights: "
        f"triplet={CUFSF_TRIPLET_WEIGHT}, "
        f"batch_hard={CUFSF_BATCH_HARD_WEIGHT}, "
        f"face_ce={CUFSF_FACE_CE_WEIGHT}, "
        f"sketch_ce={CUFSF_SKETCH_CE_WEIGHT}",
        flush=True,
    )

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device)
        print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    checkpoint_for_results = args.checkpoint
    train_samples = ""

    if not args.skip_train:
        train_dataset = FaceSketchTripletDataset(
            train_manifest_path,
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
        best_train_loss = float("inf")

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
                ],
            )
            writer.writeheader()

            for epoch in range(1, args.epochs + 1):
                if args.negative_resample_each_epoch and not isinstance(train_dataset, Subset):
                    train_dataset.samples = build_triplet_rows(train_pairs, random.Random(args.seed + epoch))

                train_stats = train_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    triplet_loss=criterion,
                    batch_hard_loss=batch_hard,
                    triplet_weight=CUFSF_TRIPLET_WEIGHT,
                    batch_hard_weight=CUFSF_BATCH_HARD_WEIGHT,
                    face_ce_weight=CUFSF_FACE_CE_WEIGHT,
                    sketch_ce_weight=CUFSF_SKETCH_CE_WEIGHT,
                    log_interval=args.log_interval,
                    accumulation_steps=args.accumulation_steps,
                    use_amp=args.amp,
                    scaler=scaler,
                    track_grad_norm=True,
                )

                if train_stats.loss < best_train_loss:
                    best_train_loss = train_stats.loss
                    save_checkpoint(best_train_path, model, epoch, best_train_loss, args, train_manifest_path, test_pairs_path)

                save_checkpoint(last_path, model, epoch, train_stats.loss, args, train_manifest_path, test_pairs_path)

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
                    }
                )
                handle.flush()

                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"train_loss={train_stats.loss:.4f} "
                    f"triplet={train_stats.triplet_loss:.4f} "
                    f"batch_hard={train_stats.batch_hard_loss:.4f} "
                    f"triplet_acc={train_stats.triplet_acc:.4f} "
                    f"grad_norm={train_stats.grad_norm:.4f}",
                    flush=True,
                )

        if last_path.exists():
            checkpoint_for_results = str(last_path)
            load_checkpoint(last_path, model, device)
            print(f"Testing with last checkpoint: {last_path}", flush=True)

    elif args.checkpoint:
        checkpoint_for_results = args.checkpoint

    result_rows = []
    failed_match_rows = []
    diagnostics_rows = []
    if not args.skip_test:
        test_result, failures, diagnostics = run_cufsf_test(model, test_pairs, args, device)
        result_rows.append(
            {
                "run_id": args.run_id,
                "dataset": "cufsf_500_694",
                "train_identities": len(train_identities),
                "test_identities": len(test_identities),
                "test_samples": test_result["samples"],
                "top1": f"{test_result['top1']:.6f}",
                "top5": f"{test_result['top5']:.6f}",
                "top10": f"{test_result['top10']:.6f}",
                "top50": f"{test_result['top50']:.6f}",
                "mrr": f"{test_result['mrr']:.6f}",
                "map": f"{test_result['map']:.6f}",
                "mean_positive_cosine": f"{test_result['mean_positive_cosine']:.6f}",
                "runtime_seconds": f"{test_result['runtime_seconds']:.2f}",
                "checkpoint": checkpoint_for_results,
            }
        )
        failed_match_rows.extend(failures)
        diagnostics_rows.extend(diagnostic_rows(args.run_id, "cufsf_500_694", diagnostics))

    write_results(results_path, result_rows)
    write_failed_matches(failures_path, failed_match_rows)
    write_diagnostics_csv(diagnostics_path, diagnostics_rows)

    total_runtime = time.perf_counter() - started_at
    print(f"cufsf_results.csv: {results_path}", flush=True)
    print(f"failed_matches.csv: {failures_path}", flush=True)
    print(f"Total runtime: {total_runtime:.2f}s", flush=True)


if __name__ == "__main__":
    main()
