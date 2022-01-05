import torch
import gc


def clear_garbage():
    torch.cuda.empty_cache()
    _ = gc.collect()
