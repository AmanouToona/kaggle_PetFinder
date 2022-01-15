import sys

# sys.path.append("../input/timm-pytorch-image-models/pytorch-image-models-master")
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore")
# general
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
import pickle
from tqdm.auto import tqdm
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import random
import gc
import cv2

gc.enable()
import glob

pd.set_option("display.max_columns", None)
from sklearn.linear_model import RidgeCV

# visualization
import matplotlib.pyplot as plt

# %matplotlib inline

# augmentation
from albumentations.pytorch import ToTensorV2
import albumentations as A

# deep learning
import timm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR,
)
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio
from PIL import Image

# from tqdm.notebook import tqdm

# tqdm.pandas()

# metrics
from sklearn.metrics import mean_squared_error


class Config:
    model_name = "swint_large224"
    base_dir = "content/drive/MyDrive/petfinder"
    data_dir = "input/petfinder-pawpularity-score/"
    meta_data_dir = "input/trainmeta/"
    model_dir = "."
    output_dir = "."
    img_train_dir = os.path.join(data_dir, "train")
    img_test_dir = os.path.join(data_dir, "test")
    random_seed = 555
    n_epoch = 5
    n_fold = 5
    tta = True  # calculate cv score in case TTA is executed
    tta_times = 4
    tta_beta = 1 / tta_times
    model_path = "swin_large_patch4_window7_224"
    pretrained = True
    inp_channels = 3
    im_size = 224
    lr = 2e-5
    opt_wd_non_norm_bias = 0.01
    opt_wd_norm_bias = 0
    opt_beta1 = 0.9
    opt_beta2 = 0.99
    opt_eps = 1e-5
    batch_size = 16
    epoch_step_valid = 3
    steps_per_epoch = 62
    num_workers = 8
    out_features = 1
    dropout = 0
    aug_decay_epoch = 4
    mixup = False
    if mixup:
        mixup_epoch = n_epoch
    else:
        mixup_epoch = 0
    mixup_alpha = 0.2
    scheduler_name = "OneCycleLR"  # OneCycleLR
    reduce_lr_factor = 0.6
    reduce_lr_patience = 1
    T_0 = 4
    T_max = 4
    T_mult = 1
    min_lr = 1e-7
    max_lr = 2e-5
    is_debug = False
    if is_debug:
        n_epoch = 1
        aug_decay_epoch = 1
        n_fold = 2
        n_sample_debug = 500
        tta_times = 2
        tta_beta = 1 / tta_times


def seed_everything(seed=Config.random_seed):
    # os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def return_imgfilepath(name, folder=""):
    # path = os.path.join(folder, f'{name}.jpg')
    path = f"input/train/{name}.jpg"
    return path




class PetNet(nn.Module):
    def __init__(
        self,
        model_name=Config.model_path,
        out_features=Config.out_features,
        inp_channels=Config.inp_channels,
        pretrained=Config.pretrained,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=inp_channels, num_classes=out_features
        )

    def forward(self, image):
        output = self.model(image)
        return output





class PetDataset(Dataset):
    def __init__(self, image_filepaths, targets, transform=None):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        with open(image_filepath, "rb") as f:
            image = Image.open(f)
            image_rgb = image.convert("RGB")
        image = np.array(image_rgb)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image / 255  # convert to 0-1
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        target = self.targets[idx]

        image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return image, target


def get_train_transforms(epoch, dim=Config.im_size):
    return A.Compose(
        [
            # resize like Resize in fastai
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.RandomCrop(height=dim, width=dim, p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
            # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    )


def get_inference_fixed_transforms(mode=0, dim=Config.im_size):
    if mode == 0:  # do not original aspects, colors and angles
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ],
            p=1.0,
        )


def get_train_transforms(epoch, dim=Config.im_size):
    return A.Compose(
        [
            # resize like Resize in fastai
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.RandomCrop(height=dim, width=dim, p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
            # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    )


def get_inference_fixed_transforms(mode=0, dim=Config.im_size):
    if mode == 0:  # do not original aspects, colors and angles
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ], p=1.0,
        )


def divice_norm_bias(model):
    norm_bias_params = []
    non_norm_bias_params = []
    except_wd_layers = ["norm", ".bias"]
    for n, p in model.model.named_parameters():
        if any([nd in n for nd in except_wd_layers]):
            norm_bias_params.append(p)
        else:
            non_norm_bias_params.append(p)
    return norm_bias_params, non_norm_bias_params


def usr_rmse_score(output, target):
    y_pred = torch.sigmoid(output).cpu()
    y_pred = y_pred.detach().numpy() * 100
    target = target.cpu() * 100

    return mean_squared_error(target, y_pred, squared=False)


def rmse_oof(_oof_df, fold=None):
    oof_df = _oof_df.copy()
    if fold is not None:
        oof_df = oof_df[oof_df["fold"] == fold]
    target = oof_df["Pawpularity"].values
    y_pred = oof_df["pred"].values
    if fold is not None:
        print(f"fold {fold}: {mean_squared_error(target, y_pred, squared=False)}")
    else:
        print(f"overall: {mean_squared_error(target, y_pred, squared=False)}")


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def get_scheduler(optimizer):
    scheduler = None
    if Config.scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, eta_min=Config.min_lr, last_epoch=-1)
    elif Config.scheduler_name == "OneCycleLR":
        # div=25
        # initial_lr =max_lr/div
        # default last_lr =initial lr / final_div_factor(10000) = max_lr
        # in case fastai  default last_lr =max_lr / div_final(100000)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=Config.max_lr,
            pct_start=0.25,  # same as fastai, defaut 0.3
            steps_per_epoch=int(((Config.n_fold - 1) * train_df.shape[0]) / (Config.n_fold * Config.batch_size)) + 1,
            epochs=Config.n_epoch,
        )
        print()

    elif Config.scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1)
    elif Config.scheduler_name == "ReduceOnPlateauLR":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=Config.reduce_lr_factor, patience=Config.reduce_lr_patience, verbose=True
        )
    return scheduler


def training_loop(filepaths, targets, train_df):
    num_bins = int(np.floor(1 + (3.3) * (np.log2(len(train_df)))))
    target_bins = pd.cut(targets, bins=num_bins, labels=False)
    # device optimization
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    skf = StratifiedKFold(n_splits=Config.n_fold, shuffle=True, random_state=Config.random_seed)
    oof_df = pd.DataFrame()

    for i_fold, (train_idx, valid_idx) in enumerate(skf.split(filepaths, target_bins)):
        print(f"=== fold {i_fold}: training ===")
        """
        separate train/valid data
        """
        X_train_paths = filepaths[train_idx]
        y_train = targets[train_idx]

        """
        prepare dataset
        """
        train_dataset = PetDataset(image_filepaths=X_train_paths, targets=y_train, transform=get_train_transforms(0))

        """
        create dataloader
        """
        train_loader = DataLoader(
            train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True
        )

        """
        instantiate model, cost function and optimizer
        """
        model = PetNet()
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        norm_bias_params, non_norm_bias_params = divice_norm_bias(model)
        # print(f"norm bias params: {len(norm_bias_params)}, non norm bias params: {len(non_norm_bias_params)}")
        optimizer = torch.optim.AdamW(
            [
                {"params": norm_bias_params, "weight_decay": Config.opt_wd_norm_bias},
                {"params": non_norm_bias_params, "weight_decay": Config.opt_wd_non_norm_bias},
            ],
            betas=(Config.opt_beta1, Config.opt_beta2),
            eps=Config.opt_eps,
            lr=Config.lr,
            amsgrad=False,
        )
        scheduler = get_scheduler(optimizer)

        """
        train / valid loop
        """

        scaler = GradScaler()
        for epoch in range(1, Config.n_epoch + 1):
            print(f"=== fold:{i_fold} epoch: {epoch}: training ===")

            metric_monitor = MetricMonitor()
            stream = tqdm(train_loader)

            for _, (images, target) in enumerate(stream, start=1):
                model.train()

                images = images.to(device, non_blocking=True).float()
                target = target.to(device, non_blocking=True).float().view(-1, 1)
                optimizer.zero_grad()

                with autocast():  # mixed precision
                    output = model(images)

                    loss = criterion(output, target)

                rmse_score = usr_rmse_score(output, target)
                metric_monitor.update("Loss", loss.item())
                metric_monitor.update("RMSE", rmse_score)
                stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if (scheduler is not None) & (Config.scheduler_name != "ReduceOnPlateauLR"):
                    scheduler.step()

        del model, output, train_loader, train_dataset
        gc.collect()

        torch.cuda.empty_cache()
    return oof_df.sort_values("Id")


if __name__ == "__main__":
    seed_everything()

    train_file_path = "input/train.csv"
    train_df = pd.read_csv(train_file_path)

    train_df["file_path"] = train_df["Id"].apply(lambda x: return_imgfilepath(x))
    train_df["norm_score"] = train_df["Pawpularity"] / 100


    num_bins = int(np.floor(1 + (3.3) * (np.log2(len(train_df)))))
    train_df["bins"] = pd.cut(train_df["norm_score"], bins=num_bins, labels=False)
    train_df["fold"] = -1

    skf = StratifiedKFold(n_splits=Config.n_fold, shuffle=True, random_state=Config.random_seed)
    for i, (_, train_index) in enumerate(skf.split(train_df.index, train_df["bins"])):
        train_df.iloc[train_index, -1] = i

    train_df["fold"] = train_df["fold"].astype("int")

    ids = train_df["Id"].values
    filepaths = train_df["file_path"].values
    targets = train_df["Pawpularity"].values / 100


    oof_df = training_loop(filepaths, targets, train_df)
