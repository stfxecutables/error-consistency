from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _pytest.capture import CaptureFixture
import torch
from numpy import ndarray
from pandas import DataFrame, Series

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy
from torch import Tensor, nn
from torch.nn import Conv2d, Linear, ReLU, Module, BatchNorm2d, Sequential, Softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from error_consistency.constants import LIB_ROOT

PT_SAVEDIR = LIB_ROOT / "error_consistency/deep"
SIZE = (32, 32)


class ConvUnit(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
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
        self.relu = ReLU()
        self.norm = BatchNorm2d(num_features=out_channels)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x  # type: ignore


class GlobalAveragePooling(Module):
    """Apply the mean along a dimension.

    Parameters
    ----------
    reduction_dim: int = 1
        The dimension along which to apply the mean.
    """

    def __init__(self, reduction_dim: Union[int, Tuple[int, ...]] = 1, keepdim: bool = False):
        super().__init__()
        self._reduction_dim = reduction_dim
        self._keepdim = keepdim

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self._reduction_dim, keepdim=self._keepdim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(reducing_dim={self._reduction_dim})"

    __repr__ = __str__


class FmnistDM(LightningDataModule):
    def __init__(self, hparams: Dict[str, Any], seed: int = None) -> None:
        super().__init__()
        self.batch_size = hparams["batch_size"]
        self.seed = seed

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # download only
        FashionMNIST(PT_SAVEDIR, train=True, download=True, transform=transforms.ToTensor())
        FashionMNIST(PT_SAVEDIR, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage: Optional[str] = None) -> None:
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        all_train = FashionMNIST(PT_SAVEDIR, train=True, download=False, transform=transform)
        test = FashionMNIST(PT_SAVEDIR, train=False, download=False, transform=transform)
        gen = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        train, val = random_split(all_train, [55000, 5000], gen)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


class FMnistCNN(LightningModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        depth = hparams["depth"]
        w = width = hparams["width"]
        layers: List[Module] = []
        for i in range(depth + 1):
            if i == 0:
                layers.append(ConvUnit(in_channels=1, out_channels=width))
                continue
            layers.append(ConvUnit(in_channels=w, out_channels=2 * w))  # double each time
            w *= 2
        layers.append(GlobalAveragePooling(reduction_dim=(2, 3)))
        self.output = Linear(w, 10)
        self.seq = Sequential(*layers)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.seq(x)
        return self.output(x)

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # training_step defined the train loop. It is independent of forward
        loss, acc = self.generic_step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, acc = self.generic_step(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, acc = self.generic_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def generic_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y = batch
        out = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        preds = out.max(axis=1)[1]
        acc = accuracy(preds, y)
        return loss, acc

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def test_prelim(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        HPARAMS: Dict = dict(batch_size=256, depth=4, width=8)
        dm = FmnistDM(HPARAMS)  # type: ignore
        model = FMnistCNN(HPARAMS)
        trainer = Trainer(gpus=1, max_epochs=1)
        trainer.fit(model, dm)  # type: ignore
        result = trainer.test()[0]
        pprint(result)
        print(f"Testing Accuracy: {result['test_acc']}")
