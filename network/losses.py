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


def classification_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean()
