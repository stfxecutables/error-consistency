import torch
from torch import Tensor
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from typing import Any, Callable, Optional, no_type_check
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Compose,
    Resize,
    ToPILImage,
)


from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, TensorDataset

DATA = Path(__file__).resolve().parent.parent.parent / "tests/datasets/covid-ct"
RESIZE = 224


class CovidDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, transform: Callable) -> None:
        super().__init__()
        self.dataset = TensorDataset(x, y)
        self.transform = transform

    def __getitem__(self, index: int) -> Tensor:
        x, y = self.dataset[index]
        return self.transform(x), y

    def __len__(self) -> int:
        return len(self.dataset)


class CovidCTDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 6) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

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

    @staticmethod
    def _get_dataset(subset: str) -> CovidDataset:
        transform = (
            Compose(
                [
                    ToPILImage(),
                    RandomResizedCrop(RESIZE, scale=(0.5, 1.0)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                ]
            )
            if subset == "train"
            else Compose([ToPILImage(), Resize(RESIZE), ToTensor()])
        )
        x = torch.from_numpy(np.load(DATA / f"x_{subset}.npy")).unsqueeze(1)
        y = torch.from_numpy(np.load(DATA / f"y_{subset}.npy")).unsqueeze(1).float()
        return CovidDataset(x, y, transform)

