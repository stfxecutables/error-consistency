import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import torch
from numpy import ndarray
from torch.nn import PReLU, Sequential
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.datamodule import trainloader_length

EPOCHS = [1000, 2000, 3000, 4000, 5000]
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
        self.total_steps = trainloader_length(batch_size) * max_epochs
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
    lr_key = f"{self.params.version}{'-pretrain' if self.params.pretrain else ''}"
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
    steps = trainloader_length(self.params.batch_size)
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
    mode = self.params.cyclic_mode
    mode_alts = {"tr": "triangular", "tr2": "triangular2", "gamma": "exp_range"}
    mode = mode_alts[mode] if mode in mode_alts else mode
    # lr_key = f"{self.params.version}{'-pretrain' if self.params.pretrain else ''}"
    # max_lr = MAX_LRS[lr_key]
    # base_lr = MIN_LRS[lr_key]
    max_lr, base_lr = self.params.cyclic_max, self.params.cyclic_base
    steps_per_epoch = trainloader_length(self.params.batch_size)
    epochs = self.params.max_epochs
    cycle_length = epochs / steps_per_epoch  # division by two makes us hit min FAST
    stepsize_up = cycle_length // 2
    # see lr_scheduling.py for motivation behind this
    r = base_lr / max_lr
    f = 60  # f == 1 means final max_lr is base_lr. f == 100 means triangular
    gamma = np.exp(np.log(f * r) / epochs) if mode == "exp_range" else 1.0
    stepsize_up = stepsize_up // 2 if mode == "exp_range" else stepsize_up
    sched = CyclicLR(
        optimizer, base_lr=base_lr, max_lr=max_lr, mode=mode, step_size_up=stepsize_up, gamma=gamma
    )
    scheduler: Dict[str, Union[CyclicLR, str]] = {"scheduler": sched, "interval": "step"}
    return [optimizer], [scheduler]


def cosine_scheduling(self: Any) -> Tuple[List[Optimizer], List[LRScheduler]]:
    # optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    # return [optimizer], [scheduler]
    raise NotImplementedError("CosineAnnealingLR not implemented yet.")


def linear_test_scheduling(self: Any) -> Tuple[List[Optimizer], List[LRScheduler]]:
    optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    lr_min = self.params.lrtest_min
    lr_max = self.params.lrtest_max
    n_epochs = self.params.lrtest_epochs_to_max
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


def test_gamma_cyclic_values() -> None:
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
        f = 60  # desired final max oscillation height times base_lr
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


def test_random_values() -> None:
    BATCH_SIZE = 128
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=len(EPOCHS), sharey=True)
    for i, epochs in enumerate(EPOCHS):
        model = Sequential(PReLU())
        trainloader_length(BATCH_SIZE)
        base_lr = 1e-4
        max_lr = 0.01
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


if __name__ == "__main__":
    # plot_decay_function(0.001)
    # plot_multiplicative_decay_function(0.01)

    # test_small_cyclic_values()
    # test_gamma_cyclic_values()
    # test_triangular_cyclic_values()
    test_random_values()
