import os
import sys
import ray
import numpy as np
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple, no_type_check
from ray.tune import report

import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.metrics.functional import accuracy
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
    linear_test_scheduling,
    onecycle_scheduling,
)
from analysis.covid.custom_layers import GlobalAveragePooling
from analysis.covid.arguments import to_namespace

SIZE = (256, 256)

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None


# NOTE: For ResNet:
# All pre-trained models expect input images normalized in the same way, i.e.
# mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are
# expected to be at least 224. The images have to be loaded in to a range of
# [0, 1] and then normalized using:
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
class CovidResNet(Module):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        version = hparams["version"]
        pre = hparams["pretrain"]
        # ResNet output is [batch_size, 1000]
        self.model = torchvision.models.resnet18(pretrained=pre)
        # self.model = torch.hub.load("pytorch/vision:v0.9.0", f"resnet{version}", pretrained=pre)
        self.output = GlobalAveragePooling()
        # in_features = self._get_output_size()
        # in_features = self._get_output_size()
        # self.linear = Linear(in_features=in_features, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
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

    def __init__(self, config: Dict[str, Any], ray=True, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = CovidResNet(config)
        self.params = config
        self.config = config
        self.lr = config["initial_lr"]
        self.weight_decay = config["weight_decay"]
        self.lr_schedule = config["lr_schedule"]
        self.ray = ray
        self.train_data: TensorDataset = self._get_dataset("train")
        self.val_data: TensorDataset = self._get_dataset("val")
        self.test_data: TensorDataset = self._get_dataset("test")

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        epoch = int(self.current_epoch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("epoch", epoch, sync_dist=True)
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
        loss = Loss()(out.unsqueeze(1), y)
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
        elif self.lr_schedule == "one-cycle":
            return self.onecycle_scheduling()
        elif self.lr_schedule == "linear-test":
            return self.linear_test_scheduling()
        else:
            return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @no_type_check
    def prepare_data(self, *args, **kwargs):
        return
        self.train: TensorDataset = self._get_dataset("train")
        self.val: TensorDataset = self._get_dataset("val")
        self.test: TensorDataset = self._get_dataset("test")

    @no_type_check
    def setup(self, stage: str) -> None:
        # see
        # https://github.com/UCSD-AI4H/COVID-CT/blob/master/baseline%20methods/DenseNet169/DenseNet_predict.py#L79-L93
        self.train_data: TensorDataset = self._get_dataset("train")
        self.val_data: TensorDataset = self._get_dataset("val")
        self.test_data: TensorDataset = self._get_dataset("test")

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
            # batch_size=int(np.min([self.batch_size, len(self.val)])),  # val set is 118 images
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
            # batch_size=int(np.min([self.batch_size, len(self.test)])),  # test set is 203 images
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

    cyclic_scheduling = cyclic_scheduling
    cosine_scheduling = cosine_scheduling
    linear_test_scheduling = linear_test_scheduling
    onecycle_scheduling = onecycle_scheduling


def callbacks(config: Dict[str, Any]) -> List[Callback]:
    logdir = ResNetArgs.info_from_args(config, info="logpath")
    return [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping("val_acc", min_delta=0.001, patience=100, mode="max"),
        ModelCheckpoint(
            dirpath=logdir,
            filename="{epoch}-{step}_{val_acc:.2f}_{train_acc:0.3f}",
            monitor="val_acc",
            save_last=True,
            save_top_k=1,
            mode="max",
            save_weights_only=False,
        ),
    ]


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


def train(config: Dict[str, Any]) -> Dict:
    dm = CovidCTDataModule(config)
    model = CovidLightningResNet(config, ray=False)
    # if you read the source, the **kwargs in Trainer.from_argparse_args just call .update on an
    # args dictionary, so you can override what you want with it
    defaults = trainer_defaults(config)
    trainer = Trainer.from_argparse_args(
        to_namespace(config), callbacks=callbacks(config), **defaults
    )
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm)
    return results[0]  # type: ignore


def get_config() -> Dict[str, Any]:
    parser = ResNetArgs.program_level_parser()
    parser = ResNetArgs.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    return hparams.__dict__  # type: ignore


if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = get_config()
    result = train(config)
    print(result)
