import os
import sys
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Tuple, no_type_check, List, Union

import torch
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    Callback,
    ModelCheckpoint,
)
from pytorch_lightning.metrics.functional import accuracy, auroc, f1
from torch import Tensor
from torch.nn import BatchNorm2d
from torch.nn import BCEWithLogitsLoss as Loss
from torch.nn import Conv2d, LeakyReLU, Module
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.lr_scheduler import (
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
    LambdaLR,
)  # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.datamodule import CovidCTDataModule, trainloader_length
from analysis.covid.transforms import RESIZE
from analysis.covid.arguments import EfficientNetArgs

SIZE = (256, 256)

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None

# max_lr as determined by the LR range test (https://arxiv.org/pdf/1708.07120.pdf)
# note these were tested with a batch size of 128 and various regularization params:
#
# efficientnet-bX-pretrained/_LINEAR-TEST_lr-max=0.05@1500_L2=1.00e-05_128batch_crop+rflip+elstic
# fmt: off
MAX_LRS: Dict[str, float] = {
    "b0": 0.01,
    "b1": 0.01,
    "b0-pretrain": 0.01,
    "b1-pretrain": 0.01,
}
# MIN_LR = 1e-4
MIN_LR = 1e-3
MIN_LRS: Dict[str, float] = {
    "b0": MIN_LR,
    "b1": MIN_LR,
    "b0-pretrain": MIN_LR,
    "b1-pretrain": MIN_LR,
}
# fmt: on


class GlobalAveragePooling(Module):
    """Apply the mean along a dimension.

    Parameters
    ----------
    reduction_dim: int = 1
        The dimension along which to apply the mean.
    """

    def __init__(self, reduction_dim: int = 0, keepdim: bool = False):
        super().__init__()
        self._reduction_dim = reduction_dim
        self._keepdim = keepdim

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self._reduction_dim, keepdim=self._keepdim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(reducing_dim={self._reduction_dim})"

    __repr__ = __str__


class ConvUnit(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 0,
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
        # self.prelu = PReLU()
        self.prelu = LeakyReLU()
        self.norm = BatchNorm2d(num_features=out_channels)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.prelu(x)
        x = self.norm(x)
        return x


class CovidEfficientNet(Module):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        # self.model = EfficientNet.from_pretrained(
        #     "efficientnet-b1", in_channels=1, num_classes=1, image_size=(RESIZE, RESIZE)
        # )

        version = hparams.version
        if hparams.pretrain is True:
            self.model = EfficientNet.from_pretrained(
                f"efficientnet-{version}", in_channels=1, num_classes=1, image_size=(RESIZE, RESIZE)
            )
        else:
            self.model = EfficientNet.from_name(
                f"efficientnet-{version}",
                in_channels=1,
                num_classes=1,
                image_size=(RESIZE, RESIZE),
                dropout_rate=hparams.dropout,
            )
        # in_features = self._get_output_size()
        # self.gap = GlobalAveragePooling(reduction_dim=1, keepdim=True)
        # in_features = self._get_output_size()
        # self.linear = Linear(in_features=in_features, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  # type: ignore


class CovidLightningEfficientNet(LightningModule):
    def __init__(self, hparams: Namespace, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = CovidEfficientNet(hparams)
        self.params = hparams
        self.lr = hparams.initial_lr
        self.weight_decay = hparams.weight_decay
        self.lr_schedule = hparams.lr_schedule

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    @no_type_check
    def step_helper(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        return loss, acc

    @no_type_check
    def configure_optimizers(self) -> Optimizer:
        # use the same as
        # https://github.com/UCSD-AI4H/COVID-CT/blob/0c83254a43230de176489a9b4e3ac12e23b0df53/
        # baseline%20methods/DenseNet169/DenseNet_predict.py#L554
        if self.lr_schedule == "cosine":
            return self.cosine_scheduling()
        elif self.lr_schedule == "cyclic":
            return self.cyclic_scheduling()
        elif self.lr_schedule == "one-cycle":
            return self.onecycle_scheduling()
        elif self.lr_schedule == "linear-test":
            return self.linear_test_scheduling()
        else:
            return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def cyclic_scheduling(self) -> Tuple[List[Optimizer], List[Dict[str, Union[CyclicLR, str]]]]:
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
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            mode=mode,
            step_size_up=stepsize_up,
            gamma=gamma,
        )
        scheduler: Dict[str, Union[CyclicLR, str]] = {"scheduler": sched, "interval": "step"}
        return [optimizer], [scheduler]

    def cosine_scheduling(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        # optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10)
        # return [optimizer], [scheduler]
        raise NotImplementedError("CosineAnnealingLR not implemented yet.")

    def linear_test_scheduling(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_min = self.params.lrtest_min
        lr_max = self.params.lrtest_max
        n_epochs = self.params.lrtest_epochs_to_max
        lr_step = lr_max / n_epochs
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: (lr_min + epoch * lr_step) / self.lr  # type: ignore
        )
        return [optimizer], [scheduler]

    def onecycle_scheduling(
        self
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[LRScheduler, str]]]]:
        # as per "Cyclical Learning Rates for Training Neural Networks" arXiv:1506.01186, and
        # "Super-Convergence: Very Fast Training of NeuralNetworks", arXiv:1708.07120, we do a
        # linear increase of the learning rate ("learning rate range test, LR range tst") for a
        # few epochs and note how accuracy changes.
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_key = f"{self.params.version}{'-pretrain' if self.params.pretrain else ''}"
        max_lr = MAX_LRS[lr_key]
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


def callbacks(hparams: Namespace) -> List[Callback]:
    logdir = EfficientNetArgs.info_from_args(hparams, info="logpath")
    return [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping("train_acc", min_delta=0.001, patience=300, mode="max"),
        ModelCheckpoint(
            dirpath=logdir,
            filename="{epoch}-{step}_{val_acc:.2f}_{train_acc:0.3f}",
            monitor="val_acc",
            save_last=True,
            save_top_k=2,
            mode="max",
            save_weights_only=False,
        ),
    ]


def trainer_defaults(hparams: Namespace) -> Dict:
    logdir = EfficientNetArgs.info_from_args(hparams, info="logpath")
    refresh_rate = 0 if IN_COMPUTE_CANADA_JOB else None
    max_epochs = 2000 if hparams.lr_schedule == "linear-test" else hparams.max_epochs
    return dict(
        default_root_dir=logdir,
        progress_bar_refresh_rate=refresh_rate,
        gpus=1,
        max_epochs=max_epochs,
        min_epochs=1000,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = EfficientNetArgs.program_level_parser()
    parser = EfficientNetArgs.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    dm = CovidCTDataModule(hparams)
    model = CovidLightningEfficientNet(hparams)
    # if you read the source, the **kwargs in Trainer.from_argparse_args just call .update on an
    # args dictionary, so you can override what you want with it
    trainer = Trainer.from_argparse_args(
        hparams, callbacks=callbacks(hparams), **trainer_defaults(hparams)
    )
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm)
    # we don't really need to print because tensorboard logs the test result
    for key, val in results[0].items():
        print(f"{key:>12}: {val:1.4f}")
