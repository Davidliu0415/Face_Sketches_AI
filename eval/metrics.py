import torch
from torch.nn import functional as F


@torch.no_grad()
def cosine_scores(face_embeddings, sketch_embeddings):
    face_embeddings = F.normalize(face_embeddings, p=2, dim=1)
    sketch_embeddings = F.normalize(sketch_embeddings, p=2, dim=1)
    return face_embeddings @ sketch_embeddings.t()


@torch.no_grad()
def top1_retrieval_accuracy(face_embeddings, sketch_embeddings, labels):
    scores = cosine_scores(face_embeddings, sketch_embeddings)
    predictions = scores.argmax(dim=1)
    return (labels[predictions] == labels).float().mean()


@torch.no_grad()
def paired_retrieval_metrics(face_embeddings, sketch_embeddings, topk=(1, 5)):
    return retrieval_metrics(face_embeddings, sketch_embeddings, topk=topk)


@torch.no_grad()
def retrieval_metrics(face_embeddings, sketch_embeddings, face_labels=None, sketch_labels=None, topk=(1, 5)):
    scores = cosine_scores(face_embeddings, sketch_embeddings)
    positive_mask = retrieval_positive_mask(scores, face_labels, sketch_labels)
    max_k = min(max(topk), scores.size(1))
    _, predictions = scores.topk(max_k, dim=1)

    metrics = {}
    for k in topk:
        actual_k = min(k, scores.size(1))
        correct = positive_mask.gather(1, predictions[:, :actual_k]).any(dim=1).float().mean()
        metrics[f"top{k}"] = correct.item()

    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    sorted_indices = scores.argsort(dim=1, descending=True)
    sorted_positive = positive_mask.gather(1, sorted_indices)
    first_positive_rank = sorted_positive.float().argmax(dim=1).float() + 1.0
    precision_at_rank = sorted_positive.cumsum(dim=1).float() / torch.arange(
        1,
        scores.size(1) + 1,
        device=scores.device,
        dtype=torch.float32,
    )
    average_precision = (precision_at_rank * sorted_positive.float()).sum(dim=1) / positive_counts.float()

    metrics["mrr"] = torch.reciprocal(first_positive_rank).mean().item()
    metrics["map"] = average_precision.mean().item()
    metrics["mean_positive_cosine"] = scores[positive_mask].mean().item()
    metrics["positive_cosine_std"] = scores[positive_mask].std(unbiased=False).item()
    metrics["first_positive_rank_mean"] = first_positive_rank.mean().item()
    return metrics


@torch.no_grad()
def retrieval_positive_mask(scores, face_labels=None, sketch_labels=None):
    if face_labels is None or sketch_labels is None:
        if scores.size(0) != scores.size(1):
            raise ValueError("Paired retrieval without labels requires equal face and sketch counts.")
        target = torch.arange(scores.size(0), device=scores.device)
        return target.view(-1, 1).eq(target.view(1, -1))

    face_labels = torch.as_tensor(face_labels, device=scores.device)
    sketch_labels = torch.as_tensor(sketch_labels, device=scores.device)
    return face_labels.view(-1, 1).eq(sketch_labels.view(1, -1))


@torch.no_grad()
def retrieval_failures(
    face_embeddings,
    sketch_embeddings,
    face_paths,
    sketch_paths,
    identities,
    run_id,
    dataset_name,
    face_labels=None,
    sketch_labels=None,
):
    scores = cosine_scores(face_embeddings, sketch_embeddings)
    positive_mask = retrieval_positive_mask(scores, face_labels, sketch_labels)
    top_scores, top_predictions = scores.max(dim=1)
    positive_scores = scores.masked_fill(~positive_mask, float("-inf")).max(dim=1).values

    failures = []
    for index, predicted_index in enumerate(top_predictions.tolist()):
        if positive_mask[index, predicted_index]:
            continue

        positive_indices = positive_mask[index].nonzero(as_tuple=False).flatten()
        correct_index = positive_indices[0].item()
        correct_score = positive_scores[index].item()
        predicted_score = top_scores[index].item()
        correct_rank = int((scores[index] > positive_scores[index]).sum().item() + 1)
        failures.append(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "identity": identities[index],
                "face_image": face_paths[index],
                "predicted_sketch": sketch_paths[predicted_index],
                "correct_sketch": sketch_paths[correct_index],
                "similarity": f"{predicted_score:.6f}",
                "correct_similarity": f"{correct_score:.6f}",
                "rank": correct_rank,
                "error_type": "",
                "notes": "",
            }
        )

    return failures


@torch.no_grad()
def retrieval_diagnostics(face_embeddings, sketch_embeddings, sketch_paths=None, topn=10):
    scores = cosine_scores(face_embeddings, sketch_embeddings)
    positive_scores = scores.diag() if scores.size(0) == scores.size(1) else torch.empty(0)
    top_scores, top_predictions = scores.max(dim=1)
    repeated = torch.bincount(top_predictions, minlength=scores.size(1))
    repeated_counts, repeated_indices = repeated.topk(min(topn, repeated.numel()))
    gap = top_scores - positive_scores if positive_scores.numel() == top_scores.numel() else torch.empty(0)

    rows = []
    for count, index in zip(repeated_counts.tolist(), repeated_indices.tolist()):
        if count <= 0:
            continue
        rows.append(
            {
                "sketch_index": index,
                "count": count,
                "share": count / max(1, scores.size(0)),
                "sketch_path": sketch_paths[index] if sketch_paths else "",
            }
        )

    diagnostics = {
        "score_mean": scores.mean().item(),
        "score_std": scores.std(unbiased=False).item(),
        "top1_similarity_mean": top_scores.mean().item(),
        "top1_similarity_std": top_scores.std(unbiased=False).item(),
        "top_repeated_predictions": rows,
    }
    if positive_scores.numel():
        diagnostics.update(
            {
                "paired_positive_cosine_mean": positive_scores.mean().item(),
                "paired_positive_cosine_std": positive_scores.std(unbiased=False).item(),
                "top1_minus_paired_positive_mean": gap.mean().item(),
                "top1_minus_paired_positive_std": gap.std(unbiased=False).item(),
            }
        )
    return diagnostics
