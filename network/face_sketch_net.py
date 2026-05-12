from torch import nn

from .heads import ArcFaceHead
from .mobilefacenet import MobileFaceNet


class FaceSketchMatcher(nn.Module):
    """Shared MobileFaceNet encoder with independent face and sketch ArcFace heads."""

    def __init__(
        self,
        embedding_size=512,
        dropout=0.1,
        face_num_classes=2,
        sketch_num_classes=2,
        arcface_scale=64.0,
        arcface_margin=0.35,
    ):
        super().__init__()
        self.backbone = MobileFaceNet(embedding_size=embedding_size, dropout=dropout)
        self.face_head = ArcFaceHead(
            in_features=embedding_size,
            out_features=face_num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )
        self.sketch_head = ArcFaceHead(
            in_features=embedding_size,
            out_features=sketch_num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def embed(self, images):
        return self.backbone(images)

    def forward(self, face=None, sketch=None, face_labels=None, sketch_labels=None):
        outputs = {}

        if face is not None:
            face_embedding = self.embed(face)
            outputs["face_embedding"] = face_embedding
            outputs["face_logits"] = self.face_head(face_embedding, face_labels)

        if sketch is not None:
            sketch_embedding = self.embed(sketch)
            outputs["sketch_embedding"] = sketch_embedding
            outputs["sketch_logits"] = self.sketch_head(sketch_embedding, sketch_labels)

        return outputs
