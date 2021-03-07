from argparse import Namespace
from typing import Callable, cast, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Rand2DElastic as Elastic
from monai.transforms import RandGaussianNoise, RandSpatialCrop, Resize
from torch import Tensor
from torchvision.transforms import Compose
from typing_extensions import Literal

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


def get_transform(
    hparams: Namespace, subset: Literal["train", "validation", "val"]
) -> Optional[Transform]:
    rcrop = RandSpatialCrop(roi_size=RESIZE, random_center=True, random_size=False)
    if hparams.no_rand_crop and subset in ["validation", "val"]:
        return None
    rflip = RandomHorizontalFlip(p=0.3, spatial_axis=-1)
    noise = RandGaussianNoise(prob=0.2)
    elast = Elastic(**ELASTIC_ARGS)
    train_transforms = list(
        filter(
            lambda t: t is not None,
            [
                rcrop if not hparams.no_rand_crop else None,
                rflip if not hparams.no_flip else None,
                noise if hparams.noise else None,
                elast if not hparams.no_elastic else None,
            ],
        )
    )
    val_transforms = [Resize(spatial_size=[RESIZE, RESIZE])]
    transform = Compose(train_transforms) if subset == "train" else Compose(val_transforms)
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
    transform = get_transform("train")  # type: ignore
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
