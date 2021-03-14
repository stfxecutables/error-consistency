from typing import List

import sys
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import seaborn as sbn
import torch
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn import Sequential, PReLU
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
from pytorch_lightning.metrics.functional import accuracy, auroc, f1
from torch.nn import BCEWithLogitsLoss as Loss
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from typing import Any, Optional, no_type_check, Tuple
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.datamodule import trainloader_length

EPOCHS = [1000, 2000, 3000, 4000, 5000]
LR0 = 0.01


def dummy_data() -> TensorDataset:
    x = torch.rand([10, 10])
    y = torch.round(torch.rand([10]))
    return TensorDataset(x, y)


def plot_decay_function(lr0: float, epochs: List[int] = EPOCHS) -> ndarray:
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    for i, epoch in enumerate(epochs):
        e = np.arange(epoch)
        lrs = lr0 / (lr0 + np.exp(1 / (epoch / 1000) * lr0 * e))
        c = i / len(epochs)
        color = (c, c, c)
        sbn.lineplot(x=e, y=lrs, color=color, label=epoch, ax=ax)
    ax.set_title(f"Initial LR={lr0}")
    plt.show()


def sig(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def plot_multiplicative_decay_function(lr0: float, epochs: List[int] = EPOCHS) -> ndarray:
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    for i, epoch in enumerate(epochs):
        e = np.arange(epoch)
        lrs = lr0 * (1 - 1e-3) ** e
        c = 1 - i / len(epochs)
        color = (c, c, c)
        # add a vertical shift for illustration only
        sbn.lineplot(x=e, y=lrs + i * lr0, color=color, alpha=sig(sig(c)), label=epoch, ax=ax)
    ax.set_title(f"Initial LR={lr0}")
    plt.show()


def test_small_cyclic_values() -> None:
    BATCH_SIZE = 128
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS))
    for i, epochs in enumerate(EPOCHS):
        model = Sequential(PReLU())
        optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
        steps_per_epoch = trainloader_length(BATCH_SIZE)
        cycle_length = epochs / (1 * steps_per_epoch)
        stepsize_up = cycle_length // 2
        max_lr = 0.01
        base_lr = 1e-4
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, mode="triangular2", step_size_up=stepsize_up
        )
        lrs = [scheduler.get_last_lr()]
        es = [0]
        for e in range(epochs):
            for step in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())
                es.append(e)
        axes.flat[i].plot(es, lrs, color="black", label="epoch")
        axes.flat[i].set_ylabel("LR")
        axes.flat[i].set_xlabel("Epoch")
        axes.flat[i].set_title(f"stepsize_up={stepsize_up}")
        axes.flat[i].legend().set_visible(True)
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4)
    plt.show()


def test_triangular_cyclic_values() -> None:
    BATCH_SIZE = 128
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS))
    for i, epochs in enumerate(EPOCHS):
        model = Sequential(PReLU())
        optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
        steps_per_epoch = trainloader_length(BATCH_SIZE)
        cycle_length = epochs / (2 * steps_per_epoch)
        stepsize_up = cycle_length // 2
        max_lr = 0.01
        base_lr = 1e-4
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, mode="triangular", step_size_up=stepsize_up
        )
        lrs = [scheduler.get_last_lr()]
        es = [0]
        for e in range(epochs):
            for step in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())
                es.append(e)
        axes.flat[i].plot(es, lrs, color="black", label="epoch")
        axes.flat[i].set_ylabel("LR")
        axes.flat[i].set_xlabel("Epoch")
        axes.flat[i].set_title(f"stepsize_up={stepsize_up}")
        axes.flat[i].legend().set_visible(True)
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4)
    plt.show()


def test_gamma_cyclic_values() -> None:
    BATCH_SIZE = 128
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS))
    for i, epochs in enumerate(EPOCHS):
        model = Sequential(PReLU())
        optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
        steps_per_epoch = trainloader_length(BATCH_SIZE)
        cycle_length = epochs / (1 * steps_per_epoch)
        stepsize_up = cycle_length // 2
        max_lr = 0.01
        base_lr = 1e-4
        # when using gamma we need to ensure at the last epoch the max is some reasonable value
        # IMO something like `base_lr * 4` is fine. Since the actual `cyc_max_lr` will be
        # `max_lr * gamma ** epoch`, we can get where we want by solving this equation. That is
        # by solving:
        #
        #       max_lr * gamma ** max_epochs = base_lr * 4
        #
        # The solution to this is:
        #
        #       gamma = np.exp(np.log(base_lr * 4 / max_lr)  / max_epochs)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            mode="exp_range",
            step_size_up=stepsize_up / 3,
            gamma=np.exp(np.log(base_lr * 40 / max_lr) / epochs),
        )
        lrs = [scheduler.get_last_lr()]
        es = [0]
        for e in range(epochs):
            for step in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())
                es.append(e)
        axes.flat[i].plot(es, lrs, color="black", label="epoch")
        axes.flat[i].set_ylabel("LR")
        axes.flat[i].set_xlabel("Epoch")
        axes.flat[i].set_title(f"stepsize_up={stepsize_up}")
        axes.flat[i].legend().set_visible(True)
    fig.set_size_inches(w=6 * len(EPOCHS), h=4)
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    plt.show()


if __name__ == "__main__":
    # plot_decay_function(0.001)
    # plot_multiplicative_decay_function(0.01)

    # test_small_cyclic_values()
    test_gamma_cyclic_values()
    # test_triangular_cyclic_values()
