import torch
from torch import nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """Cosine-distance triplet loss for face/sketch matching."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = 1.0 - F.cosine_similarity(anchor, positive)
        negative_distance = 1.0 - F.cosine_similarity(anchor, negative)
        return torch.relu(positive_distance - negative_distance + self.margin).mean()


class BatchHardTripletLoss(nn.Module):
    """Batch-hard cosine triplet loss over identity labels."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        labels = labels.view(-1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        distances = 1.0 - embeddings @ embeddings.t()

        same_identity = labels.view(-1, 1).eq(labels.view(1, -1))
        eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        positive_mask = same_identity & ~eye
        negative_mask = ~same_identity

        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return embeddings.new_zeros(())

        hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        return torch.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin).mean()


def classification_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean()
