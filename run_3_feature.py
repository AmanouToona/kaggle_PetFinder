# read config file
import argparse
import collections
import sched
from tabnanny import verbose
from scipy.sparse.construct import rand
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
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold, KFold

import gc

from sklearn.metrics import mean_squared_error

from typing import Collection, List, Dict, Tuple, Optional

from collections import defaultdict

from scipy.sparse import coo_matrix
import copy

from tqdm import tqdm

# not kaggle environment
from models import SimpleModel, SimpleModelSig, SimpleModelDrop, Repro001, NoUseMeta, NoMetaSwa, UseMeta

from timm.scheduler import CosineLRScheduler


from PIL import Image

# kernel での実行時は以下の関数とモデルをべた書きする
from utils import EarlyStopping, early_stopping, set_seed, clear_garbage, torch_faster

import joblib
from lightgbm import LGBMRegressor
import lightgbm as lgb

kaggle_kernel = False  # kaggle kernel で実行する際の設定

TRAINED_MODEL = Path("model_trained")

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


def rmse(y_true, y_pred):
    y_true = np.array(y_true) * 100
    y_pred = np.array(y_pred) * 100
    return np.sqrt(mean_squared_error(y_true, y_pred))


def usr_rmse_score(target, y_pred):
    y_pred = np.array(y_pred) * 100
    target = np.array(target) * 100
    return mean_squared_error(target, y_pred, squared=False)


# ====================================
# DataSet
# ====================================
train_meta_data = pd.read_csv(INPUT / "train.csv")
target_col = "Pawpularity"


class TrainDataset(Dataset):
    def __init__(self, df, train_mode=True, transform=Optional[List[str]]) -> None:
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


class ReproDataset(Dataset):
    def __init__(self, df, transform=None) -> None:
        super().__init__()
        self.df = df
        self.df["file_path"] = self.df.apply(get_file_path, axis=1)
        self.image_filepaths = self.df["file_path"].values
        self.targets = self.df[target_col].values
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
            for transform in self.transform:
                image = transform(image=image)["image"]

        image = image / 255  # convert to 0-1
        target = self.targets[idx]

        target = torch.tensor(target, dtype=torch.float)
        return image, target


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
        self.meta_data = train_meta_data[dense_feature]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta = self.meta_data.iloc[idx]
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
    model_weight = "final_003_fold_01.pth"
    d = torch.load(TRAINED_MODEL / model_weight, map_location=torch.device("cpu"))
    model.load_state_dict(d)
    model.to(device)

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

    # ====================================
    #  train loop
    # ====================================
    logger.debug("start train section")
    report = defaultdict(list)
    iteration = 0
    accumulation = 1
    if "accumulation" in config:
        accumulation = config["accumulation"]

    for epoch in range(config["train"]["max_epoch"]):
        logger.info("\n")
        logger.info(f'epoch {epoch + 1:02} / {int(config["train"]["max_epoch"]):02} --------------------------------')

        model.eval()
        running_loss = 0.0
        y_true_train = []
        features_train = []
        meta_train = []
        with torch.no_grad():
            for step, (images, meta, targets) in enumerate(tqdm(train_loader)):
                y_true_train.extend(targets)
                images, meta, targets = (
                    images.float().to(device, non_blocking=True),
                    meta.to(device, non_blocking=True).float(),
                    targets.to(device, non_blocking=True).float(),
                )

                y, feature, _ = model(images, meta)
                features_train.append(feature.clone().detach().cpu().numpy())
                meta_train.append(meta.clone().detach().cpu().numpy())

                del y
                del feature

            y_true_train = np.array(y_true_train)
            meta_train = np.concatenate(meta_train, axis=0)
            features_train = np.concatenate(features_train, axis=0)

        x_tr = np.concatenate([features_train, meta_train], axis=1)
        train_data = lgb.Dataset(x_tr, label=y_true_train)

        # ====================================
        #  valid loop
        # ====================================
        logger.debug("start validation section")
        model.eval()
        running_loss = 0.0
        y_true_valid = []
        features_valid = []
        meta_valid = []
        with torch.no_grad():
            for images, meta, targets in tqdm(valid_loader):
                y_true_valid.extend(targets)

                images, meta, targets = (
                    images.float().to(device, non_blocking=True),
                    meta.to(device, non_blocking=True).float(),
                    targets.to(device, non_blocking=True).float(),
                )

                y, feature, _ = model(images, meta)
                features_valid.append(feature.clone().detach().cpu().numpy())
                meta_valid.append(meta.clone().detach().cpu().numpy())

                del y
                del feature
        y_true_valid = np.array(y_true_valid)
        print()
        x_va = np.concatenate([features_valid, meta_valid], axis=1)
        valid_data = lgb.Dataset(x_va, label=y_true_valid)

        lgb_param = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "seed": 42,
            "max_depth": -1,
            "min_data_in_leaf": 10,
            "verbosity": -1,
        }

        reg = LGBMRegressor()
        reg.fit(x_tr, y_true_train)

        pred = reg.predict(x_va)

        print(x_va.shape, x_tr.shape)
        clf = lgb.train(
            lgb_param, train_data, 10000, valid_set=[train_data, valid_data], verbose_eval=10, early_stopping=10
        )

        report["valid_nn_loss"].append(running_loss / len(valid_loader))
        logger.info(f'valid loss          : {report["valid_nn_loss"][-1]:.8f}')

        if "metrics" in config.keys():
            res = calc_metrics(truths, preds, config["metrics"])

            for key, value in res.items():
                report[f"valid_{key}"].append(value)

        loss = 0  # todo
        if config["scheduler"]["name"] not in [
            "OneCycleLR",
            "CosineAnnealingWarmRestarts",
            "CosineAnnealingLR",
            "CosineLRScheduler",
        ]:
            scheduler.step()

        if config["scheduler"]["name"] == "CosineLRScheduler":
            scheduler.step(1 + epoch)

        if use_swa and epoch > config["swa"]["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # early stopping
        if use_early_stop and early_stop.step(report["valid_nn_loss"][-1]):
            break

        # save model
        # ToDo パラメータを逐次保存しもっともよいパラメータを呼び出すように変更する
        _ = gc.collect()
        torch.save(model.state_dict(), f'model_trained/{config["global"]["name"]}_epoch{epoch:02}.pth')

    torch.save(model.state_dict(), f'model_trained/{config["global"]["name"]}.pth')

    epochs = [i + 1 for i in range(len(report["train_nn_loss"]))]
    eval_df = pd.DataFrame.from_dict(report)
    eval_df.index = epochs
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
        # config["train"]["max_epoch"] = 2

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

    fold_idxs = list()
    if "fold_type" in config.keys() and config["fold_type"] == "skf":
        num_bins = int(np.floor(1 + (3.3) * (np.log2(len(train_meta_data)))))
        fold_make = train_meta_data.copy()
        fold_make["bins"] = pd.cut(fold_make[target_col], bins=num_bins, labels=False)
        fold_make["fold"] = -1

        Fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for train_idx, valid_idx in Fold.split(fold_make, fold_make["bins"]):
            fold_idxs.append((train_idx, valid_idx))
    else:
        Fold = KFold(n_splits=n_folds, shuffle=config["train"]["shuffle"], random_state=SEED)
        for train_idx, valid_idx in Fold.split(train_meta_data):
            fold_idxs.append((train_idx, valid_idx))

    del n_folds

    train_meta_data[target_col] /= 100

    # ====================================
    #  run train valid function
    # ====================================
    preds = []
    truths = []
    for fold_id, train_valid_index in enumerate(fold_idxs):
        logger.info("\n")
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
