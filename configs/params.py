import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
save = True
write_log = True

method = "MobileFaceNet-FaceSketch"
remarks = "dual_arcface_triplet"

embedding_size = 512
dropout = 0.1

face_num_classes = 2
sketch_num_classes = 2

arcface_scale = 64.0
arcface_margin = 0.35

triplet_margin = 0.5
triplet_weight = 1.0
face_ce_weight = 1.0
sketch_ce_weight = 1.0

batch_size = 16
num_workers = 4
epochs = 30
lr = 1e-3
weight_decay = 1e-5

image_size = 112
