from argparse import Namespace
from typing import Callable, cast, Optional

import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from monai.transforms import Rand2DElastic as Elastic
from monai.transforms import RandGaussianNoise, RandSpatialCrop, Resize
from torch import Tensor
from torchvision.transforms import Compose
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.arguments import EfficientNetArgs
from analysis.covid.preprocessing import NUMPY_DATA_ROOT

Transform = Callable[[Tensor], Tensor]

RESIZE = 224
ELASTIC_ARGS_DEFAULT = dict(
    spacing=0.5,
    magnitude_range=(0.01, 0.2),
    prob=1.0,
    rotate_range=np.pi / 40,  # radians
    shear_range=0.1,
    translate_range=(0.3, 0.3),
    # scale_range=(0.1, 0.1),
    scale_range=(0.2, 0.2),
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
    arg_overrides = dict(
        rotate_range=hparams.elastic_degree * np.pi / 180,
        shear_range=hparams.elastic_shear,
        translate_range=(hparams.elastic_trans, hparams.elastic_trans),
        scale_range=(hparams.elastic_scale, hparams.elastic_scale),
    )
    elast = Elastic(**{**ELASTIC_ARGS_DEFAULT, **arg_overrides})
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
    # Torch magic rescale values, e.g. in PyTorch ResNet documentation, or
    # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/26
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    hparams = EfficientNetArgs.defaults()
    arg_overrides = dict(
        rotate_range=hparams.elastic_degree * np.pi / 180,
        shear_range=hparams.elastic_shear,
        translate_range=(hparams.elastic_trans, hparams.elastic_trans),
        scale_range=(hparams.elastic_scale, hparams.elastic_scale),
    )
    args = {**ELASTIC_ARGS_DEFAULT, **arg_overrides}
    transform = Elastic(**args)
    data = np.load(str(NUMPY_DATA_ROOT / "x_test.npy"))
    idx = np.random.randint(0, len(data))
    x = np.load(str(NUMPY_DATA_ROOT / "x_test.npy"))[idx]
    if x.ndim == 2:  # BW
        x = np.expand_dims(x, 0)
    elif x.ndim == 3 and x.shape[-1] == 3:  # color
        x = x.transpose([2, 0, 1])  # monai needs (C, H, W)
        # x = x[0, :, :]
        # x = np.expand_dims(x, 0)
    fig, axes = plt.subplots(ncols=5, nrows=5)
    center = 12
    for i in range(25):
        if i == center:
            continue  # center of grid
        img = transform(x)
        if len(img.shape) == 3:
            img = img.transpose([1, 2, 0])  # imshow needs (H, W, C)
            img *= STD[np.newaxis, np.newaxis, :]
            img += MEAN[np.newaxis, np.newaxis, :]
            axes.flat[i].imshow(img)
        else:
            axes.flat[i].imshow(img.squeeze(), cmap="Greys")
        # axes.flat[i].set_title("Elastic Deformed")
    if len(x.shape) == 3:
        img = x.transpose([1, 2, 0])  # imshow needs (H, W, C)
        img *= STD[np.newaxis, np.newaxis, :]
        img += MEAN[np.newaxis, np.newaxis, :]
        axes.flat[center].imshow(img)
    else:
        axes.flat[center].imshow(img.squeeze(), cmap="Greys")
    axes.flat[center].set_title(f"Original image {idx}")
    fig.set_size_inches(w=16, h=18)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    fig.suptitle(str(args))
    plt.show()


def test_total() -> None:
    hparams = EfficientNetArgs.defaults()
    transform = get_transform(hparams, "train")  # type: ignore
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
    fig.suptitle(str(ELASTIC_ARGS_DEFAULT))
    plt.show()


if __name__ == "__main__":
    for _ in range(2):
        test_elastic()
        # test_total()
