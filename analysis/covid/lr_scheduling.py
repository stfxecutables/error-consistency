import sys
import math
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import torch
from numpy import ndarray
from torch.nn import PReLU, Sequential
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, LambdaLR, StepLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.datamodule import trainloader_length

EPOCHS = [100, 300]
LR0 = 0.01


class RandomLinearDecayingLR(_LRScheduler):
    def __init__(
        self: Any,
        optimizer: Optimizer,
        eta_min: float = 1e-6,
        eta_max: float = 0.1,
        max_epochs: int = None,
        batch_size: int = None,
        last_epoch: int = -1,
    ) -> None:
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.max_epochs = max_epochs
        # self.batch_size = batch_size
        self.total_steps = trainloader_length(batch_size)  # * max_epochs
        super().__init__(optimizer, last_epoch=last_epoch)

    @no_type_check
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        emin, emax = self.eta_min, self.eta_max
        epoch_max = emin + (emax - emin) * (1 - self.last_epoch / self.max_epochs)
        lr = float(np.random.uniform(self.eta_min, epoch_max))
        return [lr for base_lr in self.base_lrs]


def onecycle_scheduling(
    self: Any,
) -> Tuple[List[Optimizer], List[Dict[str, Union[LRScheduler, str]]]]:
    """As per "Cyclical Learning Rates for Training Neural Networks" arXiv:1506.01186, and
    "Super-Convergence: Very Fast Training of NeuralNetworks", arXiv:1708.07120, we do a
    linear increase of the learning rate ("learning rate range test, LR range tst") for a
    few epochs and note how accuracy changes."""
    optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    lr_key = f"{self.params['version']}{'-pretrain' if self.params['pretrain'] else ''}"
    max_lr = self.MAX_LRS[lr_key]
    # Below needs to be len(train_loader...) // batch_size. We also add 1 because there is
    # clearly an implementation bug somewhere. See:
    # https://discuss.pytorch.org/t/lr-scheduler-onecyclelr-causing-tried-to-step-57082-times-
    # the-specified-number-of-total-steps-is-57080/90083/5
    #
    # or
    #
    # https://forums.pytorchlightning.ai/t/ lr-scheduler-onecyclelr-valueerror-tried-to-step-x-
    # 2-times-the-specified-number-of-total-steps-is-x/259/3
    steps = trainloader_length(self.params["batch_size"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=None,
        epochs=self.params.max_epochs,
        pct_start=self.params.onecycle_pct,  # don't need large learning rate for too long
        steps_per_epoch=steps + 1,  # lightning bug
    )
    # Ensure `scheduler.step()` is called after each batch, i.e.
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1120#issuecomment-598331924
    scheduler = {"scheduler": scheduler, "interval": "step"}
    return [optimizer], [scheduler]


def cyclic_scheduling(self: Any) -> Tuple[List[Optimizer], List[Dict[str, Union[CyclicLR, str]]]]:
    # The problem with `triangular2` is it decays *way* too quickly.
    optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    mode = self.params["cyclic_mode"]
    mode_alts = {"tr": "triangular", "tr2": "triangular2", "gamma": "exp_range"}
    mode = mode_alts[mode] if mode in mode_alts else mode
    # lr_key = f"{self.params.version}{'-pretrain' if self.params.pretrain else ''}"
    # max_lr = MAX_LRS[lr_key]
    # base_lr = MIN_LRS[lr_key]
    max_lr, base_lr = self.params["lr_max"], self.params["lr_min"]
    steps_per_epoch = trainloader_length(self.params["batch_size"])
    epochs = self.params["max_epochs"]
    total_steps = float(epochs * steps_per_epoch)
    cycle_length = total_steps // 5
    stepsize = cycle_length / 2.0
    stepsize_up = int(np.max([1.0, stepsize / 5.0]))
    stepsize_down = int(stepsize)
    # cycle_length = epochs / steps_per_epoch  # division by two makes us hit min FAST
    # see lr_scheduling.py for motivation behind this
    r = base_lr / max_lr
    f = 60  # f == 1 means final max_lr is base_lr. f == 100 means triangular
    gamma = np.exp(np.log(f * r) / epochs) if mode == "exp_range" else 1.0
    # stepsize_up = stepsize_up // 2 if mode == "exp_range" else stepsize_up
    sched = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        mode=mode,
        step_size_up=stepsize_up,
        step_size_down=stepsize_down,
        gamma=gamma,
    )
    scheduler: Dict[str, Union[CyclicLR, str]] = {"scheduler": sched, "interval": "step"}
    return [optimizer], [scheduler]


def step_scheduling(self: Any) -> Tuple[List[Optimizer], List[Dict[str, Union[StepLR, str]]]]:
    optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    sched = StepLR(
        optimizer=optimizer, step_size=self.params["step_size"], gamma=self.params["gamma"]
    )
    scheduler: Dict[str, Union[StepLR, str]] = {"scheduler": sched, "interval": "epoch"}
    return [optimizer], [scheduler]


def exponential_scheduling(
    self: Any
) -> Tuple[List[Optimizer], List[Dict[str, Union[ExponentialLR, str]]]]:
    optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    sched = ExponentialLR(optimizer=optimizer, gamma=1.0 - self.params["gamma_sub"])
    scheduler: Dict[str, Union[ExponentialLR, str]] = {"scheduler": sched, "interval": "epoch"}
    return [optimizer], [scheduler]


def random_scheduling(
    self: Any
) -> Tuple[List[Optimizer], List[Dict[str, Union[RandomLinearDecayingLR, str]]]]:
    optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    sched = RandomLinearDecayingLR(
        optimizer=optimizer,
        eta_min=self.params["lr_min"],
        eta_max=self.params["lr_max"],
        max_epochs=self.params["max_epochs"],
        batch_size=self.params["batch_size"],
    )
    scheduler: Dict[str, Union[RandomLinearDecayingLR, str]] = {
        "scheduler": sched,
        "interval": "step",
    }
    return [optimizer], [scheduler]


def cosine_scheduling(self: Any) -> Tuple[List[Optimizer], List[LRScheduler]]:
    # optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    # return [optimizer], [scheduler]
    raise NotImplementedError("CosineAnnealingLR not implemented yet.")


def linear_test_scheduling(self: Any) -> Tuple[List[Optimizer], List[LRScheduler]]:
    optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    lr_min = self.params["lr_min"]
    lr_max = self.params["lr_max"]
    n_epochs = self.params["lrtest_epochs_to_max"]
    lr_step = lr_max / n_epochs
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: (lr_min + epoch * lr_step) / self.lr  # type: ignore
    )
    return [optimizer], [scheduler]


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


def plot_multiplicative_decay_function(
    lr0: float, gamma: float = 1e-3, epochs: List[int] = EPOCHS
) -> ndarray:
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    for i, epoch in enumerate(sorted(epochs)):
        e = np.arange(epoch)
        lrs = lr0 * (1 - gamma) ** e
        c = 1 - i / len(epochs)
        color = (c, c, c)
        # add a vertical shift for illustration only
        sbn.lineplot(
            x=e,
            y=lrs + (len(epochs) - 1 - i) * lr0 / 10,
            color=color,
            alpha=sig(sig(c)),
            label=epoch,
            ax=ax,
        )
    ax.set_title(f"Initial LR={lr0}, gamma={gamma}")
    plt.show()


def test_cyclic_values(
    base_lr: float = 0.01, max_lr: float = 1e-4, mode="triangular2", gamma_sub: float = 1e-3
) -> None:
    BATCH_SIZES = [4, 8, 16, 32, 64]
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS), nrows=len(BATCH_SIZES))
    for e, epochs in tqdm(enumerate(EPOCHS), total=len(EPOCHS)):
        for b, batch_size in enumerate(BATCH_SIZES):
            model = Sequential(PReLU())
            optimizer = SGD(model.parameters(), lr=base_lr, weight_decay=1e-5)
            steps_per_epoch = trainloader_length(batch_size)
            total_steps = float(epochs * steps_per_epoch)
            # cycle_length = epochs / (1 * steps_per_epoch)
            # cycle_length = steps_per_epoch * 4 * np.log(steps_per_epoch)
            cycle_length = total_steps / 5.0
            stepsize = cycle_length / 2.0
            stepsize_up = int(np.max([1.0, stepsize / 5.0]))
            stepsize_down = stepsize
            scheduler = CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                mode=mode,
                step_size_up=stepsize_up,
                step_size_down=stepsize_down,
                gamma=float(1.0 - gamma_sub),
            )
            lrs = [scheduler.get_last_lr()]  # type: ignore
            es = [0]
            for epoch in range(epochs):
                for _ in range(steps_per_epoch):
                    optimizer.step()
                    scheduler.step()
                    lrs.append(scheduler.get_last_lr())  # type: ignore
                    es.append(epoch)
            axes[b][e].plot(es, lrs, color="black", lw=0.2)
            axes[b][e].set_ylabel("LR")
            if (b == len(BATCH_SIZES) - 1) and (e == len(EPOCHS) - 1):
                axes[b][e].set_xlabel("Epoch")
            axes[b][e].set_title(f"btch={batch_size}, stepsize_up={stepsize_up}")
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4 * len(BATCH_SIZES))
    fig.subplots_adjust(hspace=0.4)
    plt.show()


def test_step_values(base_lr: float = 0.01, step_size: int = 30, gamma_sub: float = 1e-3) -> None:
    BATCH_SIZES = [4, 8, 16, 32, 64]
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS), nrows=len(BATCH_SIZES))
    for e, epochs in tqdm(enumerate(EPOCHS), total=len(EPOCHS)):
        for b, batch_size in enumerate(BATCH_SIZES):
            model = Sequential(PReLU())
            optimizer = SGD(model.parameters(), lr=base_lr, weight_decay=1e-5)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=float(1.0 - gamma_sub))
            lrs = [scheduler.get_last_lr()]  # type: ignore
            es = [0]
            for epoch in range(epochs):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())  # type: ignore
                es.append(epoch)
            axes[b][e].plot(es, lrs, color="black", lw=0.2)
            axes[b][e].set_ylabel("LR")
            if (b == len(BATCH_SIZES) - 1) and (e == len(EPOCHS) - 1):
                axes[b][e].set_xlabel("Epoch")
            min_lr = np.min(lrs)
            max_lr = np.max(lrs)
            axes[b][e].set_title(f"batch_size={batch_size}, lr=({min_lr:0.1e},{max_lr:0.1e})")
    fig.suptitle(f"base_lr={base_lr}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4 * len(BATCH_SIZES))
    fig.subplots_adjust(hspace=0.4)
    plt.show()


def test_exp_values(
    base_lr: float = 0.01, gamma_sub_min: float = 1e-3, gamma_sub_max: float = 1e-2
) -> None:
    EPOCHS = [10, 30, 50, 100, 200, 300]
    gammas = [1.0 - g for g in np.linspace(gamma_sub_min, gamma_sub_max, 8)]
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS), nrows=len(gammas), sharey=True)
    for e, epochs in tqdm(enumerate(EPOCHS), total=len(EPOCHS)):
        for g, gamma in enumerate(gammas):
            model = Sequential(PReLU())
            optimizer = SGD(model.parameters(), lr=base_lr, weight_decay=1e-5)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            lrs = [scheduler.get_last_lr()]  # type: ignore
            es = [0]
            for epoch in range(epochs):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())  # type: ignore
                es.append(epoch)
            axes[g][e].plot(es, lrs, color="black", lw=0.2)
            # axes[g][e].set_ylabel("LR")
            # axes[g][e].set_xlabel("Epoch")
            min_lr = np.min(lrs)
            axes[g][e].set_title(f"min_lr=({min_lr:0.1e})")
    fig.suptitle(f"base_lr={base_lr}")
    fig.text(x=0.5, y=0.05, s="Epochs", ha="center")  # X axis label
    fig.text(x=0.05, y=0.5, s="LR", va="center", rotation="vertical")  # y axis label
    fig.set_size_inches(w=6 * len(EPOCHS), h=4 * len(gammas) + 1)
    fig.subplots_adjust(top=0.9, bottom=0.11, left=0.11, right=0.9, hspace=0.7, wspace=0.2)
    plt.show()


def test_small_cyclic_values(mode="triangular2") -> None:
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
            optimizer, base_lr=base_lr, max_lr=max_lr, mode=mode, step_size_up=stepsize_up
        )
        lrs = [scheduler.get_last_lr()]  # type: ignore
        es = [0]
        for e in range(epochs):
            for step in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())  # type: ignore
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
        lrs = [scheduler.get_last_lr()]  # type: ignore
        es = [0]
        for e in range(epochs):
            for step in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())  # type: ignore
                es.append(e)
        axes.flat[i].plot(es, lrs, color="black", label="epoch")
        axes.flat[i].set_ylabel("LR")
        axes.flat[i].set_xlabel("Epoch")
        axes.flat[i].set_title(f"stepsize_up={stepsize_up}")
        axes.flat[i].legend().set_visible(True)
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4)
    plt.show()


def test_gamma_cyclic_values(f: int = 60) -> None:
    BATCH_SIZE = 128
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS))
    for i, epochs in enumerate(EPOCHS):
        model = Sequential(PReLU())
        optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
        steps_per_epoch = trainloader_length(BATCH_SIZE)
        cycle_length = epochs / steps_per_epoch
        stepsize_up = cycle_length // 2
        # max_lr = 0.01
        # base_lr = 1e-4
        max_lr = 0.1
        base_lr = 1e-3
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
        r = base_lr / max_lr
        # f = 60  # desired final max oscillation height times base_lr
        gamma = np.exp(np.log(f * r) / epochs)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            mode="exp_range",
            step_size_up=stepsize_up / 2,
            gamma=gamma,
        )
        lrs = [scheduler.get_last_lr()]  # type: ignore
        es = [0]
        for e in range(epochs):
            for step in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())  # type: ignore
                es.append(e)
        axes.flat[i].plot(es, lrs, color="black", label="epoch")
        axes.flat[i].set_ylabel("LR")
        axes.flat[i].set_xlabel("Epoch")
        axes.flat[i].set_title(f"stepsize_up={stepsize_up}")
        axes.flat[i].legend().set_visible(True)
    fig.set_size_inches(w=6 * len(EPOCHS), h=4)
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    plt.show()


def test_random_values(min_lr: float = 1e-4, max_lr: float = 0.1) -> None:
    BATCH_SIZE = 128
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS), sharey=True)
    for i, epochs in enumerate(EPOCHS):
        model = Sequential(PReLU())
        trainloader_length(BATCH_SIZE)
        base_lr = min_lr
        optimizer = SGD(model.parameters(), lr=base_lr, weight_decay=1e-5)
        scheduler = RandomLinearDecayingLR(
            optimizer, eta_max=0.01, eta_min=1e-4, max_epochs=epochs, batch_size=BATCH_SIZE
        )
        lrs = [scheduler.get_last_lr()]  # type: ignore
        es = [0]
        for e in range(epochs):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr())  # type: ignore
            es.append(e)
            # for step in range(steps_per_epoch):
            # optimizer.step()
            # scheduler.step()
            # lrs.append(scheduler.get_last_lr())
            # es.append(e)
        axes.flat[i].plot(es, lrs, color="black", label="epoch", lw=0.2)
        axes.flat[i].set_ylabel("LR")
        axes.flat[i].set_xlabel("Epoch")
        axes.flat[i].set_title(f"Min: {np.min(lrs):0.1e}\nMax: {np.max(lrs):0.1e}")
        axes.flat[i].legend().set_visible(True)
    fig.suptitle(f"Random learning rate. max_lr={max_lr}, base_lr={base_lr}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4)
    plt.show()


def test_multiplicative(
    base_lr: float = 0.01, max_lr: float = 1e-4, mode="triangular2", gamma_sub: float = 1e-3
) -> None:
    BATCH_SIZES = [4, 8, 16, 32, 64]
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS), nrows=len(BATCH_SIZES))
    for e, epochs in enumerate(EPOCHS):
        for b, batch_size in enumerate(BATCH_SIZES):
            model = Sequential(PReLU())
            optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
            steps_per_epoch = trainloader_length(batch_size)
            total_steps = e * steps_per_epoch
            # cycle_length = epochs / (1 * steps_per_epoch)
            # cycle_length = steps_per_epoch * 5
            cycle_length = total_steps // 10
            stepsize_up = cycle_length // 2
            scheduler = CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                mode=mode,
                step_size_up=stepsize_up,
                gamma=float(1.0 - gamma_sub),
            )
            lrs = [scheduler.get_last_lr()]  # type: ignore
            es = [0]
            for epoch in range(epochs):
                for _ in range(steps_per_epoch):
                    optimizer.step()
                    scheduler.step()
                    lrs.append(scheduler.get_last_lr())  # type: ignore
                    es.append(epoch)
            axes[b][e].plot(es, lrs, color="black")
            axes[b][e].set_ylabel("LR")
            axes[b][e].set_xlabel("Epoch")
            axes[b][e].set_title(f"btch={batch_size}, stepsize_up={stepsize_up}")
    fig.suptitle(f"max_lr={max_lr}, base_lr={base_lr}, stepsize_up={stepsize_up}")
    fig.set_size_inches(w=6 * len(EPOCHS), h=4 * len(BATCH_SIZES))
    plt.show()


if __name__ == "__main__":
    # plot_decay_function(0.001)
    # plot_multiplicative_decay_function(lr0=1e-3, gamma=5e-2, epochs=[30, 50, 100, 200])

    # test_small_cyclic_values()
    # test_gamma_cyclic_values(f=10)
    # test_triangular_cyclic_values()
    # test_random_values()
    # test_cyclic_values(base_lr=2e-5, max_lr=1e-4, mode="triangular2", gamma_sub=1e-2)
    # test_step_values(base_lr=3.8e-5, step_size=75, gamma_sub=0.5)
    # test_exp_values(1e-5, gamma_sub_min=5e-3, gamma_sub_max=1e-2)  # pretty reasonable
    test_exp_values(1e-5, gamma_sub_min=1e-3, gamma_sub_max=1e-2)  # maybe more reasonable
    # test_exp_values(1e-5, gamma_sub_min=8e-3, gamma_sub_max=2e-2)  # I like this one.

