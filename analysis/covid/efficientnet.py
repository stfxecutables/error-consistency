import matplotlib.pyplot as plt
import sys
from monai.transforms.spatial.array import Rand2DElastic
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss as Loss
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from pathlib import Path
from functools import reduce
from efficientnet_pytorch import EfficientNet

from torch.nn import (
    Module,
    Conv2d,
    PReLU,
    BatchNorm2d,
    Linear,
    ModuleList,
    Flatten,
    LeakyReLU,
    MaxPool2d,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy, auroc, f1
from typing import no_type_check

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.datamodule import CovidCTDataModule, RESIZE

SIZE = (256, 256)


class GlobalAveragePooling(Module):
    """Apply the mean along a dimension.

    Parameters
    ----------
    reduction_dim: int = 1
        The dimension along which to apply the mean.
    """

    def __init__(self, reduction_dim: int = 0, keepdim: bool = False):
        super().__init__()
        self._reduction_dim = reduction_dim
        self._keepdim = keepdim

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self._reduction_dim, keepdim=self._keepdim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(reducing_dim={self._reduction_dim})"

    __repr__ = __str__


class ConvUnit(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 0,
        padding: int = 0,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        # self.prelu = PReLU()
        self.prelu = LeakyReLU()
        self.norm = BatchNorm2d(num_features=out_channels)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.prelu(x)
        x = self.norm(x)
        return x


class CovidEfficientNet(Module):
    def __init__(self) -> None:
        super().__init__()
        # self.model = EfficientNet.from_pretrained(
        #     "efficientnet-b1", in_channels=1, num_classes=1, image_size=(RESIZE, RESIZE)
        # )
        self.model = EfficientNet.from_pretrained(
            "efficientnet-b0", in_channels=1, num_classes=1, image_size=(RESIZE, RESIZE)
        )
        # self.model = EfficientNet.from_name(
        #     "efficientnet-b0", in_channels=1, num_classes=1, image_size=(RESIZE, RESIZE)
        # )
        # in_features = self._get_output_size()
        # self.gap = GlobalAveragePooling(reduction_dim=1, keepdim=True)
        # in_features = self._get_output_size()
        # self.linear = Linear(in_features=in_features, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class CovidLightningEfficientNet(LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_schedule: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = CovidEfficientNet()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        # f1score = f1(pred, y, 2)
        # self.log("train_f1", f1score, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        # pred = torch.round(torch.sigmoid(out))
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        acc01 = accuracy(pred, y_int, threshold=0.1)
        acc09 = accuracy(pred, y_int, threshold=0.9)
        # auc = auroc(pred, y)
        # f1score = f1(pred, y, num_classes=2)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_auc", auc)
        # self.log("val_f1", f1score)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_acc0.10", acc01, prog_bar=False)
        self.log("val_acc0.90", acc09, prog_bar=False)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        # pred = torch.round(torch.sigmoid(out))
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        acc01 = accuracy(pred, y_int, threshold=0.1)
        acc09 = accuracy(pred, y_int, threshold=0.9)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_acc0.10", acc01)
        self.log("test_acc0.90", acc09)

    @no_type_check
    def configure_optimizers(self) -> Optimizer:
        # use the same as
        # https://github.com/UCSD-AI4H/COVID-CT/blob/0c83254a43230de176489a9b4e3ac12e23b0df53/
        # baseline%20methods/DenseNet169/DenseNet_predict.py#L554
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return [optimizer], [scheduler]
        return optimizer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    LOGDIR = str(Path(__file__).resolve().parent)
    model = CovidLightningEfficientNet()
    dm = CovidCTDataModule(batch_size=40, num_workers=6)
    # trainer = Trainer(gpus=1, val_check_interval=0.5, max_epochs=1000, overfit_batches=0.1)
    # trainer = Trainer(gpus=1, val_check_interval=0.5, max_epochs=1000)
    trainer = Trainer(default_root_dir=LOGDIR, gpus=1, max_epochs=3000)
    trainer.fit(model, datamodule=dm)

