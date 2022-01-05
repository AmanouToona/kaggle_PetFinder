import torch


def torch_faster():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
