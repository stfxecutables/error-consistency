import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from numpy import ndarray
from PIL import Image
from scipy.stats import linregress
from tqdm import tqdm
from typing_extensions import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
IMG_DATA_ROOT = DATA_ROOT / "images"
NUMPY_DATA_ROOT = PROJECT_ROOT / "tests/datasets/covid-ct"
if not NUMPY_DATA_ROOT.exists():
    os.makedirs(NUMPY_DATA_ROOT, exist_ok=True)

SIZE = (256, 256)


def get_lines(path: Path) -> List[Path]:
    parent = IMG_DATA_ROOT / "covid" if "CT_COVID" in path.name else IMG_DATA_ROOT / "control"
    contents = None
    with open(path, "r") as file:
        contents = [parent / line.replace("\n", "") for line in file.readlines()]
    return contents


# fmt: off
COVID_FILES: Dict[str, List[Path]] = {
    "val":   get_lines(DATA_ROOT / "Data-split/COVID/valCT_COVID.txt"),  # noqa
    "test":  get_lines(DATA_ROOT / "Data-split/COVID/testCT_COVID.txt"),  # noqa
    "train": get_lines(DATA_ROOT / "Data-split/COVID/trainCT_COVID.txt"),  # noqa
}
CONTROL_FILES: Dict[str, List[Path]] = {
    "val":   get_lines(DATA_ROOT / "Data-split/NonCOVID/valCT_NonCOVID.txt"), # noqa
    "test":  get_lines(DATA_ROOT / "Data-split/NonCOVID/testCT_NonCOVID.txt"), # noqa
    "train": get_lines(DATA_ROOT / "Data-split/NonCOVID/trainCT_NonCOVID.txt"), # noqa
}
# fmt: on


def inspect_sizes() -> None:
    COVID_IMGS = {}
    CONTROL_IMGS = {}
    HEIGHTS, WIDTHS = [], []
    for subset, files in COVID_FILES.items():
        imgs = [Image.open(file) for file in tqdm(files, total=len(files))]
        for img in imgs:
            HEIGHTS.append(img.size[1])
            WIDTHS.append(img.size[0])
        COVID_IMGS[subset] = imgs
    for subset, files in CONTROL_FILES.items():
        imgs = [Image.open(file) for file in tqdm(files, total=len(files))]
        for img in imgs:
            HEIGHTS.append(img.size[1])
            WIDTHS.append(img.size[0])
        CONTROL_IMGS[subset] = imgs

    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    sbn.scatterplot(x=HEIGHTS, y=WIDTHS, ax=ax)
    ax.set_xlabel("Height (px)")
    ax.set_ylabel("Width (px)")
    m, b = linregress(HEIGHTS, WIDTHS)[0:2]
    ave = np.mean(np.array(WIDTHS) / np.array(HEIGHTS))
    ax.set_title(f"Width = {m:1.3f}*Height + {b:1.3f}\nAverage W/H: {ave:1.2f}")
    print(f"Unique widths: {np.unique(WIDTHS)}")
    print(f"Unique heights: {np.unique(HEIGHTS)}")
    plt.show()


def get_numpy(file: Path, resnet: bool = False) -> ndarray:
    # all this trickery is just to not lose data
    if resnet:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR).astype(float)
    else:
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(float)
    resized = cv2.resize(img, SIZE, interpolation=cv2.INTER_LANCZOS4)
    arr = np.asarray(resized)
    return arr


def convert_subset(
    subset: Literal["train", "val", "test"], resnet: bool = False
) -> Tuple[ndarray, ndarray]:
    size = (len(COVID_FILES[subset]) + len(CONTROL_FILES[subset]), *SIZE)
    if resnet:
        size = (*size, 3)
    x = np.full(size, np.nan, dtype=np.float32)
    y = np.concatenate(
        [
            np.zeros([len(CONTROL_FILES[subset])], dtype=np.uint8).ravel(),
            np.ones([len(COVID_FILES[subset])], dtype=np.uint8).ravel(),
        ]
    )
    i = 0
    pbar = tqdm(desc=subset, total=size[0])
    for file in CONTROL_FILES[subset]:
        x[i, :, :] = get_numpy(file, resnet)
        i += 1
        pbar.update()
    for file in COVID_FILES[subset]:
        x[i, :, :] = get_numpy(file, resnet)
        i += 1
        pbar.update()
    pbar.close()
    assert np.sum(np.isnan(x)) == 0
    assert len(y) == size[0]
    return x, y


def to_numpy(normalization: str = "featurewise") -> Tuple[ndarray, ...]:
    """NOTE from PyTorch https://pytorch.org/hub/pytorch_vision_resnet/

    All pre-trained models expect input images normalized in the same way, i.e.
    mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at
    least 224. **The images have to be loaded in to a range of [0, 1]** and then normalized using
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """
    resnet = normalization == "resnet"
    x_train, y_train = convert_subset("train", resnet)
    x_val, y_val = convert_subset("val", resnet)
    x_test, y_test = convert_subset("test", resnet)
    feature_means = np.mean(x_train, axis=0, keepdims=True)
    feature_sds = np.std(x_train, axis=0, ddof=1, keepdims=True)
    x_train -= feature_means
    x_train /= feature_sds
    x_val -= feature_means
    x_val /= feature_sds
    x_test -= feature_means
    x_test /= feature_sds
    if normalization == "featurewise":
        return x_train, y_train, x_val, y_val, x_test, y_test

    # after this if we do x = np.concatenate([x_train, x_val, x_test], axis=0), we get:
    # x.max() = 3.24, x.min() = -6.49, np.median(x) = 0.105
    # np.round(np.percentile(x.ravel(), [1, 99]), 3) = [-2.332,  1.953]
    # This suggests a clip to [-2.5, 2.0] is pretty lossless, and will allow sane re-scaling
    if normalization == "resnet":
        # normalization constants from https://pytorch.org/hub/pytorch_vision_resnet/
        MIN, MAX = -2.5, 2.0
        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])
        np.clip(x_train, MIN, MAX, out=x_train)
        np.clip(x_val, MIN, MAX, out=x_val)
        np.clip(x_test, MIN, MAX, out=x_test)
        x_train = (x_train - MIN) / (MAX - MIN)
        x_val = (x_val - MIN) / (MAX - MIN)
        x_test = (x_test - MIN) / (MAX - MIN)
        # Now again if we do x = np.concatenate([x_train, x_val, x_test], axis=0), we get:
        # np.round(np.mean(x, axis=(0, 1, 2)), 3)        == [0.343, 0.343, 0.343] = mean
        # np.round(np.std(x, axis=(0, 1, 2), ddof=1), 3) == [0.299, 0.299, 0.299] = sd

        # np.newaxis allows us to broadcast as we want efficiently
        # https://stackoverflow.com/questions/7140738/numpy-divide-along-axis
        x_train -= MEAN[np.newaxis, np.newaxis, :]
        x_val -= MEAN[np.newaxis, np.newaxis, :]
        x_test -= MEAN[np.newaxis, np.newaxis, :]
        x_train /= STD[np.newaxis, np.newaxis, :]
        x_val /= STD[np.newaxis, np.newaxis, :]
        x_test /= STD[np.newaxis, np.newaxis, :]
        # to channels first
        x_train = x_train.transpose([0, 3, 1, 2])
        x_val = x_val.transpose([0, 3, 1, 2])
        x_test = x_test.transpose([0, 3, 1, 2])
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        raise ValueError("`normalization` must be one of 'resnet' or 'featurewise'.")


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = to_numpy("resnet")
    # x_train, y_train, x_val, y_val, x_test, y_test = to_numpy("featurewise")
    np.save(NUMPY_DATA_ROOT / "x_train.npy", x_train)
    np.save(NUMPY_DATA_ROOT / "y_train.npy", y_train)
    np.save(NUMPY_DATA_ROOT / "x_val.npy", x_val)
    np.save(NUMPY_DATA_ROOT / "y_val.npy", y_val)
    np.save(NUMPY_DATA_ROOT / "x_test.npy", x_test)
    np.save(NUMPY_DATA_ROOT / "y_test.npy", y_test)
    print(f"Saved normalized images to {NUMPY_DATA_ROOT}.")
