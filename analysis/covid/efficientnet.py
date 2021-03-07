import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Tuple, no_type_check

import torch
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.metrics.functional import accuracy, auroc, f1
from torch import Tensor
from torch.nn import BatchNorm2d
from torch.nn import BCEWithLogitsLoss as Loss
from torch.nn import Conv2d, LeakyReLU, Module
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.datamodule import CovidCTDataModule, trainloader_length
from analysis.covid.transforms import RESIZE

SIZE = (256, 256)

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None

# max_lr as determined by the LR range test (https://arxiv.org/pdf/1708.07120.pdf)
# note these were tested with a batch size of 128 and various regularization params:
#
# efficientnet-bX-pretrained/_LINEAR-TEST_lr-max=0.05@1500_L2=1.00e-05_128batch_crop+rflip+elstic
# fmt: off
MAX_LRS: Dict[str, float] = {
    "b0": 0.02,
    "b1": 0.02,
    "b0-pretrain": 0.01,
    "b1-pretrain": 0.01,
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
                f"efficientnet-{version}", in_channels=1, num_classes=1, image_size=(RESIZE, RESIZE)
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
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        # f1score = f1(pred, y, 2)
        # self.log("train_f1", f1score, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        # pred = torch.round(torch.sigmoid(out))
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        acc01 = accuracy(pred, y_int, threshold=0.1)
        acc09 = accuracy(pred, y_int, threshold=0.9)
        # auc = auroc(pred, y)
        # f1score = f1(pred, y, num_classes=2)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_auc", auc)
        # self.log("val_f1", f1score)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_acc0.10", acc01, prog_bar=False)
        self.log("val_acc0.90", acc09, prog_bar=False)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        out = self(x)  # out of linear layer
        loss = Loss()(out, y)
        # pred = torch.round(torch.sigmoid(out))
        pred = torch.sigmoid(out)
        y_int = y.int()
        acc = accuracy(pred, y_int)
        acc01 = accuracy(pred, y_int, threshold=0.1)
        acc09 = accuracy(pred, y_int, threshold=0.9)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_acc0.10", acc01)
        self.log("test_acc0.90", acc09)

    @no_type_check
    def configure_optimizers(self) -> Optimizer:
        # use the same as
        # https://github.com/UCSD-AI4H/COVID-CT/blob/0c83254a43230de176489a9b4e3ac12e23b0df53/
        # baseline%20methods/DenseNet169/DenseNet_predict.py#L554
        if self.lr_schedule == "cosine":
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return [optimizer], [scheduler]
        elif self.lr_schedule == "cyclic":
            optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lr_key = f"{self.params.version}{'-pretrain' if self.params.pretrain else ''}"
            max_lr = MAX_LRS[lr_key]
            base_lr = max_lr / 3.5

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr, mode="triangular2", step_size_up=200
            )
            scheduler = {"scheduler": scheduler, "interval": "step"}
            return [optimizer], [scheduler]
        elif self.lr_schedule == "one-cycle":
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lr_key = f"{self.params.version}{'-pretrain' if self.params.pretrain else ''}"
            max_lr = MAX_LRS[lr_key]
            base_lr = max_lr / 3.5
            steps = trainloader_length(self.params.batch_size)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=None,
                epochs=self.params.max_epochs,
                pct_start=self.params.onecycle_pct,  # don't need the learning rate to be large for too long
                # below needs to be len(train_loader...) // batch_size
                # we also add 1 because there is clearly an implementation bug somewhere
                # https://discuss.pytorch.org/t/lr-scheduler-onecyclelr-causing-tried-to-step-57082-times-the-specified-number-of-total-steps-is-57080/90083/5
                # or https://forums.pytorchlightning.ai/t/lr-scheduler-onecyclelr-valueerror-tried-to-step-x-2-times-the-specified-number-of-total-steps-is-x/259/3
                steps_per_epoch=steps + 1,  # lightning bug
            )
            # to understand below, see
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/1120#issuecomment-598331924
            # we need to ensure `scheduler.step()` is called after each batch
            scheduler = {"scheduler": scheduler, "interval": "step"}
            return [optimizer], [scheduler]
        # as per "Cyclical Learning Rates for Training Neural Networks" arXiv:1506.01186, and
        # "Super-Convergence: Very Fast Training of NeuralNetworks", arXiv:1708.07120, we do a
        # linear increase of the learning rate ("learning rate range test, LR range tst") for a
        # few epochs and note how accuracy changes.
        elif self.lr_schedule == "linear-test":
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lr_min = self.params.lrtest_min
            lr_max = self.params.lrtest_max
            n_epochs = self.params.lrtest_epochs_to_max
            lr_step = lr_max / n_epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: (lr_min + epoch * lr_step) / self.lr
            )
            return [optimizer], [scheduler]
        else:
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        # trainer args
        # model specific args (hparams)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--pretrain", action="store_true")  # i.e. do pre-train if flag
        parser.add_argument("--initial-lr", type=float, default=0.001)
        parser.add_argument("--weight-decay", type=float, default=0.00001)
        parser.add_argument(
            "--lr-schedule",
            choices=["cosine", "cyclic", "linear-test", "one-cycle", "none", "None"],
        )
        parser.add_argument("--onecycle-pct", type=float, default=0.05)
        parser.add_argument("--lrtest-min", type=float, default=1e-6)
        parser.add_argument("--lrtest-max", type=float, default=0.05)
        parser.add_argument("--lrtest-epochs-to-max", type=float, default=1500)

        # augmentation params
        parser.add_argument("--no-elastic", action="store_true")
        parser.add_argument("--no-rand-crop", action="store_true")
        parser.add_argument("--no-flip", action="store_true")
        parser.add_argument("--noise", action="store_true")
        return parser


def path_from_hparams(hparams: Namespace) -> str:
    hp = hparams
    ver = hp.version
    pre = "-pretrained" if hp.pretrain else ""

    # learning rate-related
    sched = hp.lr_schedule
    if sched == "one-cycle":
        sched = f"{sched}{hp.onecycle_pct:1.2f}"
    is_range_test = sched == "linear-test"
    lr = f"lr0={hp.initial_lr:1.2e}"
    if is_range_test:
        sched = str(sched).upper()
        lr = f"lr-max={hp.lrtest_max}@{hp.lrtest_epochs_to_max}"
    wd = f"L2={hp.weight_decay:1.2e}"
    b = hp.batch_size
    e = hp.max_epochs

    # augments
    crop = "crop" if not hp.no_rand_crop else ""
    flip = "rflip" if not hp.no_flip else ""
    elas = "elstic" if not hp.no_elastic else ""
    noise = "noise" if hp.noise else ""
    augs = f"{crop}+{flip}+{elas}+{noise}".replace("++", "+")
    if augs[-1] == "+":
        augs = augs[:-1]

    version_dir = Path(__file__).resolve().parent / f"logs/efficientnet-{ver}{pre}/{sched}"
    dirname = f"{lr}_{wd}_{b}batch_{e}ep_{augs}"
    log_path = str(version_dir / dirname)
    return log_path


def program_level_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, choices=[f"b{i}" for i in range(8)], default="b0")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=5000)

    return parser


def trainer_defaults(hparams: Namespace) -> Dict:
    logdir = path_from_hparams(hparams)
    refresh_rate = 0 if IN_COMPUTE_CANADA_JOB else None

    max_epochs = 2000 if hparams.lr_schedule == "linear-test" else hparams.max_epochs
    return dict(
        default_root_dir=logdir,
        progress_bar_refresh_rate=refresh_rate,
        gpus=1,
        max_epochs=max_epochs,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = program_level_parser()
    parser = CovidLightningEfficientNet.add_model_specific_args(parser)
    # line below lets
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    dm = CovidCTDataModule(hparams)
    model = CovidLightningEfficientNet(hparams)
    # if you read the source, the **kwargs in Trainer.from_argparse_args just call .update on an
    # args dictionary, so you can override what you want with it
    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    trainer = Trainer.from_argparse_args(hparams, callbacks=callbacks, **trainer_defaults(hparams))
    # trainer = Trainer(gpus=1, val_check_interval=0.5, max_epochs=1000, overfit_batches=0.1)
    # trainer = Trainer(gpus=1, val_check_interval=0.5, max_epochs=1000)
    # trainer = Trainer(default_root_dir=LOGDIR, gpus=1, max_epochs=3000)
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm)
    # we don't really need to print because tensorboard logs the test result
    for key, val in results[0].items():
        print(f"{key:>12}: {val:1.4f}")

