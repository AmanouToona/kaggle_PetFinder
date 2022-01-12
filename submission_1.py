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
from torch.utils.data import DataLoader, Dataset

# augmentation
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


import gc

from typing import List, Optional

import timm
from torch import nn

from tqdm import tqdm

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

sample_submission = pd.read_csv(INPUT / "sample_submission.csv")
meta_data = pd.read_csv(INPUT / "test.csv")
model_weight = "UseMeta_001_fold_01.pth"

# ==========================================
# utils
# ==========================================


def clear_garbage():
    torch.cuda.empty_cache()
    _ = gc.collect()


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
  name: TrainDatasetMeta

loader:
  train: {batch_size: 32, shuffle: True, num_workers: 2, pin_memory: True, drop_last: True}
  valid: {batch_size: 256, shuffle: False, num_workers: 2, pin_memory: True, drop_last: False}

model:
  name: PawpularMetaModel
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

accumulation: 2

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


class TrainDatasetMeta(Dataset):
    def __init__(self, df, train_mode=True, transform=Optional[List[str]]) -> None:
        super().__init__()
        self.df = df
        self.df["file_path"] = self.df.apply(get_file_path, axis=1)
        self.file_names = self.df["file_path"].values
        self.train_mode = train_mode
        self.transform = transform
        dense_feature = [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]
        self.meta = meta_data[dense_feature]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta = self.meta.iloc[idx]
        meta = torch.tensor(meta, dtype=torch.float)

        if self.transform:
            for transform in self.transform:
                img = transform(image=img)["image"]

        if self.train_mode:
            label = self.df[target_col]
            label = torch.tensor(label.iloc[idx]).float()
            return img, meta, label

        return img, meta


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


# ====================================
# model
# ====================================


class PawpularMetaModel(nn.Module):
    def __init__(
        self,
        base_model: str = "resnet50d",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.backbone = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels)
            final_in_features = self.backbone.fc.in_features
            if base_model == "swin_tiny_patch4_window7_224":
                final_in_features = self.backbone.head.out_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(final_in_features, 128)
        self.dense2 = nn.Linear(140, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 1)

        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(final_in_features, fc_dim))

    def forward(self, img, meta):
        x = self.backbone(img)
        feature = self.dense1(x)

        x = torch.cat([feature, meta], dim=1)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)

        return output, feature, meta


def main():
    # transforms / augmentation -------------------------------------------------------------
    submission_data = sample_submission.copy()
    if "augmentation" in config.keys():
        transforms_valid = augmentation_setter(config["augmentation"]["valid"])
    else:
        transforms_valid = []

    # data set
    dataset = config["dataset"]["name"]
    dataset = eval(dataset)(df=sample_submission, train_mode=False, transform=transforms_valid)

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
        for step, (img, meta) in enumerate(tqdm(valid_loader)):

            img = img.float().to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True).float()

            y, feature, _ = model(img, meta)

            preds.extend(y.to("cpu").numpy())

            img.detach()
            del img

    preds = np.clip(preds, 0, 100)


if __name__ == "__main__":
    main()
