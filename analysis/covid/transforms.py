import torch
from monai.transforms import Rand2DElastic as Elastic, RandSpatialCrop, RandFlip, Resize
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import (
    Compose,
    ToPILImage,
    ToTensor,
    RandomCrop,
    # RandomHorizontalFlip,
    # Resize,
)
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor

Transform = Callable[[Tensor], Tensor]

RESIZE = 224
ELASTIC_ARGS = dict(
    spacing=0.5,
    magnitude_range=(0.01, 0.2),
    prob=1.0,
    rotate_range=np.pi / 40,  # radians
    shear_range=0.1,
    translate_range=(0.3, 0.3),
    scale_range=(0.1, 0.1),
    padding_mode="reflection",
)

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.1, spatial_axis: int = 1) -> None:
        self.p = p
        self.spatial_axis = spatial_axis

    def __call__(self, x: Tensor) -> Tensor:
        flip_axis = self.spatial_axis + 1
        if np.random.uniform() < self.p:
            return torch.flip(x, dims=(flip_axis,))
        else:
            return x


def get_transform(subset: Literal["train", "validation", "val"]) -> Transform:
    transform = (
        Compose(
            [
                RandSpatialCrop(roi_size=RESIZE, random_center=True, random_size=False),
                # RandomCrop(RESIZE),
                # RandomResizedCrop(RESIZE, scale=(0.5, 1.0)),
                # RandomHorizontalFlip(),
                RandomHorizontalFlip(p=0.3, spatial_axis=-1),
                # ToTensor(),
                Elastic(**ELASTIC_ARGS),
            ]
        )
        if subset == "train"
        # else Compose([ToPILImage(), Resize(RESIZE), ToTensor()])
        # else Compose([Resize(spatial_size=[RESIZE, RESIZE]), ToTensor()])
        else Compose([Resize(spatial_size=[RESIZE, RESIZE])])
    )
    return cast(Transform, transform)


def test_elastic() -> None:
    transform = Elastic(**ELASTIC_ARGS)
    data = np.load("/home/derek/Desktop/error-consistency/tests/datasets/covid-ct/x_test.npy")
    idx = np.random.randint(0, len(data))
    x = np.load("/home/derek/Desktop/error-consistency/tests/datasets/covid-ct/x_test.npy")[idx]
    x = np.expand_dims(x, 0)
    fig, axes = plt.subplots(ncols=5, nrows=5)
    center = 12
    for i in range(25):
        if i == center:
            continue  # center of grid
        img = transform(x)
        axes.flat[i].imshow(img.squeeze(), cmap="Greys")
        # axes.flat[i].set_title("Elastic Deformed")
    axes.flat[center].imshow(x.squeeze(), cmap="Greys")
    axes.flat[center].set_title(f"Original image {idx}")
    fig.set_size_inches(w=16, h=18)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    fig.suptitle(str(ELASTIC_ARGS))
    plt.show()


def test_total() -> None:
    transform = get_transform("train")
    data = np.load("/home/derek/Desktop/error-consistency/tests/datasets/covid-ct/x_test.npy")
    idx = np.random.randint(0, len(data))
    x = np.load("/home/derek/Desktop/error-consistency/tests/datasets/covid-ct/x_test.npy")[idx]
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)
    fig, axes = plt.subplots(ncols=5, nrows=5)
    center = 12
    for i in range(25):
        if i == center:
            continue  # center of grid
        img = transform(Tensor(x))
        axes.flat[i].imshow(img.squeeze(), cmap="Greys")
        # axes.flat[i].set_title("Elastic Deformed")
    axes.flat[center].imshow(x.squeeze(), cmap="Greys")
    axes.flat[center].set_title(f"Original image {idx}")
    fig.set_size_inches(w=16, h=18)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    fig.suptitle(str(ELASTIC_ARGS))
    plt.show()


if __name__ == "__main__":
    for _ in range(5):
        # test_elastic()
        test_total()
