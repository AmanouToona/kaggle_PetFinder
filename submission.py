# read config file
import yaml

import numpy as np
import pandas as pd

# logger
import logging

# path
from pathlib import Path

# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler

# SWA  Stochastic Weight Averaging
from torch.optim.swa_utils import AveragedModel, SWALR

# augmentation
import cv2

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2

import gc

from typing import List, Dict, Tuple, Optional

import copy

from tqdm import tqdm

# not kaggle environment
# from models import SimpleModel

from PIL import Image
import torchvision

# kernel での実行時は以下の関数とモデルをべた書きする
# from utils import EarlyStopping, set_seed, clear_garbage, torch_faster

# ==========================================
# logger
# ==========================================
logger_r = logging.getLogger()
logger_r.setLevel(logging.DEBUG)
handler_st = logging.StreamHandler()
handler_st.setLevel(logging.DEBUG)
handler_st_format = logging.Formatter("%(asctime)s %(name)s: %(message)s")
handler_st.setFormatter(handler_st_format)
logger_r.addHandler(handler_st)

logger = logging.getLogger(__name__)


# ==========================================
# Path
# ==========================================
kaggle_kernel = False  # kaggle kernel で実行する際の設定

if kaggle_kernel:
    ROOT = Path.cwd().parent
    INPUT = Path("../input/petfinder-pawpularity-score")
    TRAINED_MODEL = Path("../input/petfinder-20220114")
else:
    ROOT = Path.cwd()
    INPUT = ROOT / "input"
    OUTPUT = ROOT / "output"
    LOG = ROOT / "log"
    CONFIG = ROOT / "config"
    TRAINED_MODEL = Path("model_trained")

meta_data = pd.read_csv(INPUT / "sample_submission.csv")
model_weight = "implementing01_fold_00.pth"

# ==========================================
# config
# ==========================================
config = """
# implementing batchsampler
global:
  debug: false
  seed: 42
  device: cuda

amp: true

train:
  max_epoch: 6
  fold: 5
  shuffle: true

augmentation:
  train:
    Resize: { height: 256, width: 256 }
    Normalize: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    ToTensor: {always_apply: True}
  valid:
    Resize: {height: 256, width: 256}
    Normalize: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    ToTensor: {always_apply: True}

dataset:
  name: TrainDataset

loader:
  train: {batch_size: 256, shuffle: True, num_workers: 2, pin_memory: True, drop_last: True}
  valid: {batch_size: 256, shuffle: False, num_workers: 2, pin_memory: True, drop_last: False}

model:
  name: SimpleModel
  params:
    pretrained: true
    in_channels: 3
    fc_dim: 1

optimizer:
  name: Adam
  params:
    lr: 1.0e-3

loss: MSELoss

scheduler:
  name: CosineAnnealingWarmRestarts
  params:
    T_0: 6
    T_mult: 1
    eta_min: 1.0e-7

accumulation: 1

metrics:
  - mean_squared_error
"""

config = yaml.safe_load(config)
device = torch.device(config["global"]["device"])

# ====================================
# DataSet
# ====================================
target_col = "Pawpularity"


def get_file_path(s: pd.Series):
    return INPUT / "test" / f"{s['Id']}.jpg"  # test, train で異なるパスを返すのに関数名から分からないことが美しくない


class TrainDataset(Dataset):
    def __init__(self, df, train_mode=True, transform=None) -> None:
        super().__init__()
        self.df = df
        self.df["file_path"] = self.df.apply(get_file_path, axis=1)
        self.file_names = self.df["file_path"].values
        self.train_mode = train_mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            for transform in self.transform:
                img = transform(image=img)["image"]

        if self.train_mode:
            label = self.df[target_col]
            label = torch.tensor(label.iloc[idx]).float()
            return img, label
        return img


# ====================================
# transforms / augmentation
# ====================================
class ToTensor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, image: np.array) -> torch.tensor:
        return ToTensorV2()(image=image)


# ====================================
# train valid function
# ====================================
def augmentation_setter(augmentation_list):
    augmentation = []
    for aug_method, aug_params in augmentation_list.items():
        if hasattr(A, aug_method):
            augmentation.append(getattr(A, aug_method)(**aug_params))
        else:
            try:
                augmentation.append(eval(aug_method)(**aug_params))
            except NotImplementedError:
                raise NotImplementedError
    return augmentation


def calc_metrics(truths, preds, metrics):
    res = dict()
    for metric in metrics:
        res[metric] = eval(metric)(truths, preds)
        logger.info(f"oof {metric:15}: {res[metric]:3.3f}")
    return res


def main():
    # transforms / augmentation -------------------------------------------------------------
    if "augmentation" in config.keys():
        transforms_valid = augmentation_setter(config["augmentation"]["valid"])
    else:
        transforms_valid = []

    # data set
    dataset = config["dataset"]["name"]
    dataset = eval(dataset)(df=meta_data, train_mode=False, transform=transforms_valid)

    # data loader
    valid_loader = DataLoader(dataset, **config["loader"]["valid"])

    # model
    d = torch.load(TRAINED_MODEL / model_weight, map_location=torch.device("cpu"))
    model = eval(config["model"]["name"])(**config["model"]["params"])
    model.load_state_dict(d)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i, img in enumerate(tqdm(valid_loader)):

            img = img.float().to(device, non_blocking=True)
            y = model(img).view(-1)

            preds.extend(y.to("cpu").numpy())

            img.detach()
            del img

    preds = np.clip(preds, 0, 100)


if __name__ == "__main__":
    main()
