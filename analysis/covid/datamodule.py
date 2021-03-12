import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, no_type_check, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.transforms import get_transform

DATA = Path(__file__).resolve().parent.parent.parent / "tests/datasets/covid-ct"


class CovidDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, transform: Optional[Callable]) -> None:
        super().__init__()
        self.dataset = TensorDataset(x, y)
        self.transform = transform

    def __getitem__(self, index: int) -> Tensor:
        x, y = self.dataset[index]
        if self.transform is None:
            return x, y
        return self.transform(x), y

    def __len__(self) -> int:
        return len(self.dataset)


class CovidCTDataModule(LightningDataModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.hparams = hparams

    @no_type_check
    def prepare_data(self, *args, **kwargs):
        pass

    @no_type_check
    def setup(self, stage: str) -> None:
        # see https://github.com/UCSD-AI4H/COVID-CT/blob/master/baseline%20methods/DenseNet169/DenseNet_predict.py#L79-L93
        if stage == "fit" or stage is None:
            self.train: TensorDataset = self._get_dataset("train")
            self.val: TensorDataset = self._get_dataset("val")
        if stage == "test" or stage is None:
            self.test: TensorDataset = self._get_dataset("test")

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def _get_dataset(self, subset: str) -> CovidDataset:
        transform = get_transform(self.hparams, subset)
        x = torch.from_numpy(np.load(DATA / f"x_{subset}.npy")).unsqueeze(1)
        y = torch.from_numpy(np.load(DATA / f"y_{subset}.npy")).unsqueeze(1).float()
        return CovidDataset(x, y, transform)


def trainloader_length(batch_size: int) -> int:
    """Return the number of steps in an epoch"""
    n_samples = torch.from_numpy(np.load(DATA / f"x_train.npy")).unsqueeze(1).shape[0]
    return n_samples // batch_size

