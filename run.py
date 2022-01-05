# read config file
import argparse
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

import sys

# augmentation
import cv2

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

import gc

from typing import List, Dict, Tuple, Optional

import joblib

from scipy.sparse import coo_matrix
import copy

from tqdm import tqdm

# not kaggle environment
from models import SimpleModel, SimpleModelSig

from PIL import Image
import torchvision

import copy

# kernel での実行時は以下の関数とモデルをべた書きする
from utils import EarlyStopping, set_seed, clear_garbage, torch_faster

kaggle_kernel = False  # kaggle kernel で実行する際の設定

# ====================================
# Path
# ====================================
if kaggle_kernel:
    ROOT = Path.cwd().parent
else:
    ROOT = Path.cwd()

INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
LOG = ROOT / "log"
CONFIG = ROOT / "config"

# ====================================
# logger
# ====================================
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# ====================================
# Util
# ====================================


def get_file_path(s: pd.Series):
    return INPUT / "train" / f"{s['Id']}.jpg"


# ====================================
# DataSet
# ====================================
train_meta_data = pd.read_csv(INPUT / "train.csv")
target_col = "Pawpularity"


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


class GradCAMDatset(Dataset):
    def __init__(self, df, transform) -> None:
        super().__init__()
        self.df = df
        self.img_ids = self.df["Id"].values
        self.df["file_path"] = self.df.apply(get_file_path, axis=1)
        self.file_name = self.df["file_path"].values
        self.label = self.df[target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ToDo
        return super().__getitem__(idx)


# ====================================
# transforms / augmentation
# ====================================
class ToTensor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, image: np.array) -> torch.tensor:
        return ToTensorV2()(image=image)


def mixup(data: torch.tensor, target: torch.tensor, alpha: float):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    target = target * lam + shuffled_target * (1 - lam)

    return data, target


# ====================================
# sampler
# ====================================
def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, train_meta_data, sample_size: int = 5, **kwargs):
        self.sample_size = sample_size

        if "batch_size" in kwargs.keys():
            self.batch_size = kwargs["batch_size"]
        else:
            self.batch_size = 32

        if "drop_last" in kwargs.keys():
            self.drop_last = kwargs["drop_last"]
        else:
            self.drop_last = False

        # get class nums
        df = train_meta_data.copy()
        df.reset_index(inplace=True, drop=True)
        idx_df = df.groupby("landmark_id").apply(lambda x: x.index.tolist())
        self.idx_dic = idx_df.to_dict()
        del df
        _ = gc.collect()

        # to get len
        indices = []
        for _, value in self.idx_dic.items():
            if len(value) >= self.sample_size:
                indices.extend(np.random.choice(value, self.sample_size))
            else:
                indices.extend(value)

        indices = chunk(indices, self.batch_size)
        indices = [batch.tolist() for batch in indices]

        if self.drop_last:
            if len(indices[-1]) < self.batch_size:
                indices = indices[:-2]
        self.len = len(indices)

    def __iter__(self):
        indices = []
        for _, value in self.idx_dic.items():
            if len(value) >= self.sample_size:
                indices.extend(np.random.choice(value, self.sample_size))
            else:
                indices.extend(value)

        np.random.shuffle(indices)
        indices = chunk(indices, self.batch_size)
        indices = [batch.tolist() for batch in indices]

        if self.drop_last:
            if len(indices[-1]) < self.batch_size:
                indices = indices[:-2]

        return iter(indices)

    def __len__(self):
        return self.len


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


def train_fn(config, meta_data):
    torch.backends.cudnn.benchmark = True
    device = torch.device(config["global"]["device"])

    # transforms / augmentation -------------------------------------------------------------
    if "augmentation" in config.keys():
        transforms_train = augmentation_setter(config["augmentation"]["train"])
        transforms_valid = augmentation_setter(config["augmentation"]["valid"])
    else:
        transforms_train = []
        transforms_valid = []

    if "MixUp" in config.keys():
        MIXUP = True
        alpha = config["MixUp"]["alpha"]
        r = config["MixUp"]["r"]
    else:
        MIXUP = False

    # batch sampler -------------------------------------------------------
    if "batch_sampler" in config.keys():
        batch_sampler = BalancedBatchSampler(
            meta_data.iloc[config["train"]["train_idx"]],
            sample_size=config["batch_sampler"]["sample_size"],
            **config["loader"]["train"],
        )
    else:
        batch_sampler = None

    # datasaet -------------------------------------------------------------
    logger.debug("preparation dataset")
    train_idx = config["train"]["train_idx"]
    valid_idx = config["train"]["valid_idx"]

    if "valid_resample" in config["dataset"].keys():
        # valid data のみ resampling を行い軽くする
        valid_meta_data = stratified_resample(
            n=config["dataset"]["valid_resample"]["n"], df=meta_data.iloc[valid_idx], replace=False
        )
    else:
        valid_meta_data = None

    dataset = config["dataset"]["name"]
    train_dataset = eval(dataset)(df=meta_data.iloc[train_idx], transform=transforms_train)
    if valid_meta_data is not None:
        valid_dataset = eval(dataset)(df=valid_meta_data, transform=transforms_valid)
    else:
        valid_dataset = eval(dataset)(df=meta_data.iloc[valid_idx], transform=transforms_valid)

    logger.debug(f"train dataset: {len(train_dataset):7}")
    logger.debug(f"valid dataset: {len(valid_dataset):7}")

    # data loader -------------------------------------------------------------
    logger.debug("preparation loader")
    if batch_sampler is not None:
        config["loader"]["train"]["batch_sampler"] = batch_sampler
        config["loader"]["train"]["batch_size"] = 1
        config["loader"]["train"]["shuffle"] = False
        config["loader"]["train"]["sampler"] = None
        config["loader"]["train"]["drop_last"] = False
    train_loader = DataLoader(train_dataset, **config["loader"]["train"])
    valid_loader = DataLoader(valid_dataset, **config["loader"]["valid"])

    # model -------------------------------------------------------------
    model = eval(config["model"]["name"])(**config["model"]["params"])
    model.to(device)

    # optimizer -------------------------------------------------------------
    logger.debug("preparation optimizer")
    optimizer = getattr(torch.optim, config["optimizer"]["name"])(model.parameters(), **config["optimizer"]["params"])

    if "swa" in config.keys():
        use_swa = True
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer=optimizer, swa_lr=0.05)
    else:
        use_swa = False

    # scheduler -------------------------------------------------------------
    logger.debug("preparation scheduler")
    if config["scheduler"]["name"] == "OneCycleLR":
        config["scheduler"]["params"]["total_steps"] = (
            int(np.ceil(len(train_loader) / config["accumulation"])) * config["train"]["max_epoch"]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **config["scheduler"]["params"])
    elif config["scheduler"]["name"] == "LinearIncrease":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10 ** (0.4 * epoch - 5))
    else:
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["name"])(
            optimizer, **config["scheduler"]["params"]
        )

    # loss
    logger.debug("preparation loss")
    if hasattr(nn, config["loss"]):
        loss_func = getattr(nn, config["loss"])()
    else:
        raise NotImplementedError
    loss_func.to(device)

    # Early stopping
    use_early_stop = False
    if "early_stopping" in config.keys():
        use_early_stop = True
        early_stop = EarlyStopping(**config["early_stopping"]["params"])

    # amp
    if "amp" in config.keys():
        use_amp = config["amp"]
    else:
        use_amp = False

    # 高速化
    torch_faster()

    # ====================================
    #  train loop
    # ====================================
    logger.debug("start train section")
    train_losses = []
    valid_losses = []
    learning_rates = []
    iteration = 0
    accumulation = 1
    if "accumulation" in config:
        accumulation = config["accumulation"]

    for epoch in range(config["train"]["max_epoch"]):
        logger.info(f'epoch {epoch + 1:02} / {int(config["train"]["max_epoch"]):02} --------------------------------')

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for step, (images, targets) in enumerate(tqdm(train_loader)):
            images, targets = (
                images.float().to(device, non_blocking=True),
                targets.to(device, non_blocking=True).float(),
            )

            if MIXUP and np.random.rand(1) < r:
                images, targets = mixup(images, targets, alpha)

            iteration += 1

            with torch.cuda.amp.autocast(enabled=use_amp):
                y = model(images).view(-1)
                loss = loss_func(y, targets)
                loss /= accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * accumulation

            images.detach()
            targets.detach()
            del images
            del targets
            del loss  # 計算グラフの削除によるメモリ節約
            clear_garbage()

            if (step + 1) % accumulation == 0 or (step + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                clear_garbage()

                if config["scheduler"]["name"] == "OneCycleLR":
                    scheduler.step()

        train_losses.append(running_loss / len(train_loader))
        logger.info(f'lr                  : {optimizer.param_groups[0]["lr"]}')
        logger.info(f"train loss          : {train_losses[-1]:.8f}")
        logger.info(f"iteration           : {iteration}")
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # ====================================
        #  valid loop
        # ====================================
        logger.debug("start validation section")
        model.eval()
        running_loss = 0.0
        preds = []
        truths = []
        with torch.no_grad():
            for images, targets in tqdm(valid_loader):
                images, targets = (
                    images.float().to(device, non_blocking=True),
                    targets.to(device, non_blocking=True).float(),
                )
                y = model(images).view(-1)

                loss = loss_func(y, targets).detach()
                running_loss += float(loss.detach())
                preds.extend(y.to("cpu").numpy())
                truths.extend(targets.to("cpu").numpy())

                images.detach()
                targets.detach()
                del images
                del targets
                del loss  # 計算グラフの削除によるメモリ節約

                clear_garbage()

        valid_losses.append(running_loss / len(valid_loader))
        logger.info(f"valid loss          : {valid_losses[-1]:.8f}")
        if "metrics" in config.keys():
            calc_metrics(truths, preds, config["metrics"])

        loss = 0  # todo
        if config["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler.step(loss)
        elif config["scheduler"]["name"] == "OneCycleLR":
            pass
        else:
            scheduler.step()

        if use_swa and epoch > config["swa"]["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # early stopping
        if use_early_stop and early_stop.step(valid_losses[-1]):
            break

        # save model
        # ToDo パラメータを逐次保存しもっともよいパラメータを呼び出すように変更する
        _ = gc.collect()
        torch.save(model.state_dict(), f'model_trained/{config["global"]["name"]}_epoch{epoch:02}.pth')

    torch.save(model.state_dict(), f'model_trained/{config["global"]["name"]}.pth')

    epochs = [i + 1 for i in range(len(train_losses))]
    eval_df = pd.DataFrame(index=epochs, columns=["train_eval", "valid_eval", "learning_rate"])
    eval_df["train_eval"] = train_losses
    eval_df["valid_eval"] = valid_losses
    eval_df["learning_rate"] = learning_rates
    eval_df.to_csv(OUTPUT / f'{config["global"]["name"]}_eval.csv')

    return preds


# ====================================
#  resample
# ====================================
def simple_resample(r: float, df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(r, replace=True)
    return df


def stratified_resample(n: int, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
    res = []
    for class_num in df["landmark_id"].unique():
        if replace:
            res.append(df[df["landmark_id"] == class_num].sample(n, replace=True))
        else:
            if len(df[df["landmark_id"] == class_num]) >= n:
                res.append(df[df["landmark_id"] == class_num].sample(n))
            else:
                res.append(df[df["landmark_id"] == class_num])
    res = pd.concat(res, axis=0)
    return res


def read_config(file_name: str):
    logger.info(f"reading {file_name}")

    config_file = CONFIG / f"{file_name}.yml"

    try:
        with open(str(config_file), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"{file_name} con not find")
        sys.exit()

    if "name" not in config["global"].keys():
        config["global"]["name"] = file_name

    return config


# ====================================
#  main
# ====================================
def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="pytorch runner")
    parser.add_argument("-c", "--config_file", help="runtime configuration file", default="implementing01")
    args = parser.parse_args()

    config = read_config(args.config_file)

    DEBUG = config["global"]["debug"]
    if DEBUG:
        print("!" * 10, "debug mode", "!" * 10)
        config["train"]["max_epoch"] = 2

    SEED = config["global"]["seed"]
    set_seed(SEED)

    # ====================================
    #  logging
    # ====================================
    handler_st = logging.StreamHandler()
    handler_st.setLevel(logging.DEBUG)
    handler_st_format = logging.Formatter("%(asctime)s %(name)s: %(message)s")
    handler_st.setFormatter(handler_st_format)
    logger.addHandler(handler_st)

    log_file = LOG / f'log_{config["global"]["name"]}.log'
    handler_f = logging.FileHandler(log_file, "a")
    handler_f.setLevel(logging.DEBUG)
    handler_f_format = logging.Formatter("%(asctime)s %(name)s: %(message)s")
    handler_f.setFormatter(handler_f_format)
    logger.addHandler(handler_f)

    # ====================================
    #  resample
    # ====================================
    train_meta_data = pd.read_csv(INPUT / "train.csv")

    if DEBUG:
        logger.info("DEBUG mode... random sampling meta data")
        train_meta_data = train_meta_data.sample(frac=0.01, random_state=SEED)

    if "resample" in config.keys():
        logger.info("resample meta data.")
        resample_method = eval(config["resample"]["method"])
        resample_params = config["resample"]["params"]
        train_meta_data = resample_method(df=train_meta_data, **resample_params)

    # ====================================
    #  make fold data
    # ====================================
    n_folds = config["train"]["fold"]
    Fold = StratifiedKFold(n_splits=n_folds, shuffle=config["train"]["shuffle"], random_state=SEED)
    Fold = KFold(n_splits=n_folds, shuffle=config["train"]["shuffle"], random_state=SEED)
    del n_folds

    # train_valid_indexs = list()
    # for i, (train_index, valid_index) in enumerate(Fold.split(train_meta_data, train_meta_data["landmark_id"])):
    #     train_valid_indexs.append((train_index, valid_index))

    fold_idxs = list()
    for train_idx, valid_idx in Fold.split(train_meta_data):
        fold_idxs.append((train_idx, valid_idx))

    # ====================================
    #  run train valid function
    # ====================================
    preds = []
    truths = []
    for fold_id, train_valid_index in enumerate(fold_idxs):
        logger.info(f"start train fold: {fold_id:02}")
        fold_config = copy.deepcopy(config)
        fold_config["train"]["train_idx"] = train_valid_index[0]
        fold_config["train"]["valid_idx"] = train_valid_index[1]
        fold_config["global"]["name"] += f"_fold_{fold_id:02}"

        pred = train_fn(fold_config, train_meta_data)
        preds.append(pred)
        truths.append(train_meta_data.iloc[train_valid_index[1]][target_col].values)

        del fold_config
        clear_garbage()

        if DEBUG:
            break

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    if "metrics" in config.keys():
        calc_metrics(truths, preds, config["metrics"])


if __name__ == "__main__":
    main()
