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
