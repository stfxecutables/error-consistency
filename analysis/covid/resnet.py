from __future__ import annotations

from tqdm import tqdm
import json
import os
import sys
import numpy as np
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple, no_type_check
from warnings import warn
from ray.tune import report

import torch
import torchvision
from torch.nn import Linear, Dropout
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.core.saving import load_hparams_from_yaml
from torch import Tensor
from torch.nn import BCEWithLogitsLoss as Loss
from torch.nn import Module
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.arguments import ResNetArgs
from analysis.covid.datamodule import CovidCTDataModule, get_transform, CovidDataset, DATA
from analysis.covid.lr_scheduling import (
    cosine_scheduling,
    cyclic_scheduling,
    exponential_scheduling,
    linear_test_scheduling,
    onecycle_scheduling,
    random_scheduling,
    step_scheduling,
)
from analysis.covid.custom_layers import GlobalAveragePooling
from analysis.covid.arguments import to_namespace

SIZE = (256, 256)

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None

RESNETS = {
    18: torchvision.models.resnet18,
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}


# NOTE: For ResNet:
# All pre-trained models expect input images normalized in the same way, i.e.
# mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are
# expected to be at least 224. The images have to be loaded in to a range of
# [0, 1] and then normalized using:
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
class CovidResNet(Module):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.model = RESNETS[hparams["version"]](pretrained=hparams["pretrain"])
        self.drop = Dropout(p=hparams["dropout"], inplace=True)
        output = hparams["output"]
        self.output_type = output
        if output == "gap":
            self.output = GlobalAveragePooling()
        elif output == "linear":
            self.output = Linear(1000, 1)  # ResNet outputs [batch_size, 1000]

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.drop(x)
        if self.output_type == "linear":
            x = x.squeeze()
        return self.output(x)  # type: ignore


class CovidLightningResNet(LightningModule):
    # max_lr as determined by the LR range test (https://arxiv.org/pdf/1708.07120.pdf)
    # note these were tested with a batch size of 128 and various regularization params:
    #
    # efficientnet-bX-pretrained/_LINEAR-TEST_lr-max=0.05@1500_L2=1.00e-05_128batch_crop+rflip+elstic
    # fmt: off
    MAX_LRS: Dict[str, float] = {
        "b0": 0.01,
        "b1": 0.01,
        "b0-pretrain": 0.01,
        "b1-pretrain": 0.01,
    }
    # MIN_LR = 1e-4
    MIN_LR = 1e-3
    MIN_LRS: Dict[str, float] = {
        "b0": MIN_LR,
        "b1": MIN_LR,
        "b0-pretrain": MIN_LR,
        "b1-pretrain": MIN_LR,
    }
    # fmt: on

    def __init__(self, config: Dict[str, Any], use_ray: bool = True, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = CovidResNet(config)
        self.params = config
        self.config = config
        self.lr = config["initial_lr"]
        self.weight_decay = config["weight_decay"]
        self.lr_schedule = config["lr_schedule"]
        self.ray = use_ray
        # self.train_data: TensorDataset = self._get_dataset("train")
        # self.val_data: TensorDataset = self._get_dataset("val")
        # self.test_data: TensorDataset = self._get_dataset("test")

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        epoch = int(self.current_epoch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_epoch", epoch, sync_dist=True)
        # report(training_iteration=batch_idx, val_acc=acc, val_loss=loss)
        if self.ray:
            report(
                training_iteration=int(batch_idx),
                epoch=epoch,
                val_acc=acc.clone().detach().cpu().numpy(),
                val_loss=loss.clone().detach().cpu().numpy(),
            )

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        # x, y = batch
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        if self.ray:
            report(
                test_acc=acc.clone().detach().cpu().numpy(),
                test_loss=loss.clone().detach().cpu().numpy(),
            )

    @no_type_check
    def step_helper(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        x = x.squeeze(1)
        out = self(x)  # out of GAP layer
        if self.model.output_type == "gap":
            loss = Loss()(out.unsqueeze(1), y)
        elif self.model.output_type == "linear":
            loss = Loss()(out.squeeze(), y.squeeze())
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        return loss, acc

    @no_type_check
    def configure_optimizers(self) -> Optimizer:
        # use the same as
        # https://github.com/UCSD-AI4H/COVID-CT/blob/0c83254a43230de176489a9b4e3ac12e23b0df53/
        # baseline%20methods/DenseNet169/DenseNet_predict.py#L554
        if self.lr_schedule == "cosine":
            return self.cosine_scheduling()
        elif self.lr_schedule == "cyclic":
            return self.cyclic_scheduling()
        elif self.lr_schedule == "exp":
            return self.exponential_scheduling()
        elif self.lr_schedule == "linear-test":
            return self.linear_test_scheduling()
        elif self.lr_schedule == "one-cycle":
            return self.onecycle_scheduling()
        elif self.lr_schedule == "random":
            raise NotImplementedError()
            return self.random_scheduling()
        elif self.lr_schedule == "step":
            return self.step_scheduling()
        else:
            return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @no_type_check
    def prepare_data(self, *args, **kwargs):
        self.train_data: TensorDataset = self._get_dataset("train")
        self.val_data: TensorDataset = self._get_dataset("val")
        self.test_data: TensorDataset = self._get_dataset("test")

    @no_type_check
    def setup(self, stage: str) -> None:
        # see
        self.train_data = self._get_dataset("train")
        self.val_data = self._get_dataset("val")
        self.test_data = self._get_dataset("test")

    @no_type_check
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.config["batch_size"],  # len(self.train) == 425
            num_workers=self.config["num_workers"],
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )

    @no_type_check
    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
        )

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
    #     self.log("ptl/val_loss", avg_loss)
    #     self.log("ptl/val_acc", avg_acc)

    @no_type_check
    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=1,  # test set is 203 images
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
        )

    def _get_dataset(self, subset: str) -> CovidDataset:
        transform = get_transform(self.config, subset)
        x = torch.from_numpy(np.load(DATA / f"x_{subset}.npy")).unsqueeze(1)
        y = torch.from_numpy(np.load(DATA / f"y_{subset}.npy")).unsqueeze(1).float()
        if subset == "train":
            print("Training number of samples:", len(x))
        return CovidDataset(x, y, transform)

    @staticmethod
    def from_ray_id(ray_dir: Path, trial_id: str) -> CovidLightningResNet:
        runs = sorted(ray_dir.rglob(f"*{trial_id}*"))
        if len(runs) > 1:
            raise RuntimeError("Too many matching runs.")
        elif len(runs) < 1:
            raise RuntimeError("No matching runs.")
        else:
            print(f"Found run at: {runs[0]}")
        run_dir = runs[0]
        jsonfile = run_dir / "params.json"
        config: Dict[str, Any]
        with open(jsonfile, "r") as file:
            config = json.load(file)
        defaults = trainer_defaults(config)
        dm = CovidCTDataModule(config)
        model = CovidLightningResNet(config, use_ray=False)
        trainer = Trainer.from_argparse_args(to_namespace(config), **defaults)

    @classmethod
    def from_lightning_logs(
        cls, log_dir: Path = None, version: int = 18, num: int = 1
    ) -> Tuple[List[CovidLightningResNet], List[Dict[str, Any]]]:
        def acc(path: Path) -> float:
            fname = path.name
            accuracy = float(re.match(r".*val_acc=(.*?)_", fname).group(1))
            step = int(re.match(r".*step=(.*?)_", fname).group(1))
            epoch = int(re.match(r"epoch=(.*?)-", fname).group(1))
            if epoch < 20:
                return 0
            if step < 200:
                return 0
            if accuracy > 0.99:
                return 0.0
            return accuracy

        if log_dir is None:
            log_dir = Path(__file__).resolve().parent / "logs"
        rglob = f"ResNet{version}*/**/*.ckpt"
        ckpts = sorted(
            filter(lambda p: "last" not in p.name, log_dir.rglob(rglob)), key=acc, reverse=True
        )
        best = ckpts[:num]
        ckpt_data = [torch.load(ckpt) for ckpt in best]
        configs = [ckpt_dict[cls.CHECKPOINT_HYPER_PARAMS_KEY] for ckpt_dict in ckpt_data]
        for config in configs:
            if "ray" in config:  # for some reason this messes things up
                del config["ray"]
        mdls = []
        cfgs = []
        for ckpt, config in tqdm(
            zip(ckpt_data, configs), total=len(configs), desc="Loading models"
        ):
            try:
                mdls.append(cls._load_model_state(ckpt, use_ray=False))
                cfgs.append(config["config"])
            except (TypeError, KeyError):
                pass
        # mdls = [cls._load_model_state(ckpt, use_ray=False) for ckpt in ckpt_data]
        # cfgs = [config["config"] for config in configs]  # parent key
        return mdls, cfgs

    cyclic_scheduling = cyclic_scheduling
    cosine_scheduling = cosine_scheduling
    linear_test_scheduling = linear_test_scheduling
    onecycle_scheduling = onecycle_scheduling
    random_scheduling = random_scheduling
    step_scheduling = step_scheduling
    exponential_scheduling = exponential_scheduling


def callbacks(config: Dict[str, Any]) -> List[Callback]:
    logdir = ResNetArgs.info_from_args(config, info="logpath")
    cbs = [
        LearningRateMonitor(logging_interval="epoch") if config["lr_schedule"] else None,
        EarlyStopping("val_acc", min_delta=0.001, patience=100, mode="max"),
        ModelCheckpoint(
            dirpath=logdir,
            filename="{epoch}-{step}_{val_acc:.2f}_{train_acc:0.3f}",
            monitor="val_acc",
            save_last=True,
            save_top_k=3,
            mode="max",
            save_weights_only=False,
        ),
    ]
    return list(filter(lambda c: c is not None, cbs))  # type: ignore


def trainer_defaults(config: Dict[str, Any]) -> Dict:
    logdir = ResNetArgs.info_from_args(config, info="logpath")
    refresh_rate = 0 if IN_COMPUTE_CANADA_JOB else None
    max_epochs = 2000 if config["lr_schedule"] == "linear-test" else config["max_epochs"]
    return dict(
        default_root_dir=logdir,
        progress_bar_refresh_rate=refresh_rate,
        gpus=1,
        max_epochs=max_epochs,
        min_epochs=30,
    )


def get_config() -> Dict[str, Any]:
    parser = ResNetArgs.program_level_parser()
    parser = ResNetArgs.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    return hparams.__dict__  # type: ignore


def train() -> Dict:
    config = get_config()
    dm = CovidCTDataModule(config)
    model = CovidLightningResNet(config, use_ray=False)
    # if you read the source, the **kwargs in Trainer.from_argparse_args just call .update on an
    # args dictionary, so you can override what you want with it
    defaults = trainer_defaults(config)
    trainer = Trainer.from_argparse_args(
        to_namespace(config), callbacks=callbacks(config), **defaults
    )
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm)
    return results[0]  # type: ignore


if __name__ == "__main__":
    torch.cuda.empty_cache()
    result = train()
    print(result)
