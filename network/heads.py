import math

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


class ArcFaceHead(nn.Module):
    """ArcFace classification head.

    When labels are omitted, the head returns scaled cosine logits for inference.
    """

    def __init__(self, in_features, out_features=2, scale=64.0, margin=0.35, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-9, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scale={self.scale}, margin={self.margin}"
        )
