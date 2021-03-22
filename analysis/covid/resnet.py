import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple, no_type_check

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.metrics.functional import accuracy
from torch import Tensor
from torch.nn import BCEWithLogitsLoss as Loss
from torch.nn import Module
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.arguments import ResNetArgs
from analysis.covid.datamodule import CovidCTDataModule
from analysis.covid.lr_scheduling import (
    cosine_scheduling,
    cyclic_scheduling,
    linear_test_scheduling,
    onecycle_scheduling,
)
from analysis.covid.custom_layers import GlobalAveragePooling

SIZE = (256, 256)

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None


# NOTE: For ResNet:
# All pre-trained models expect input images normalized in the same way, i.e.
# mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are
# expected to be at least 224. The images have to be loaded in to a range of
# [0, 1] and then normalized using:
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
class CovidResNet(Module):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        version = hparams.version
        pre = hparams.pretrain
        # ResNet output is [batch_size, 1000]
        self.model = torchvision.models.resnet18(pretrained=pre)
        # self.model = torch.hub.load("pytorch/vision:v0.9.0", f"resnet{version}", pretrained=pre)
        self.output = GlobalAveragePooling()
        # in_features = self._get_output_size()
        # in_features = self._get_output_size()
        # self.linear = Linear(in_features=in_features, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return self.output(x)  # type: ignore


class CovidLightningResNet(LightningModule):
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

    def __init__(self, hparams: Namespace, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = CovidResNet(hparams)
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
        # x, y = batch
        loss, acc = self.step_helper(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    @no_type_check
    def step_helper(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        x = x.squeeze(1)
        out = self(x)  # out of GAP layer
        loss = Loss()(out.unsqueeze(1), y)
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

    cyclic_scheduling = cyclic_scheduling
    cosine_scheduling = cosine_scheduling
    linear_test_scheduling = linear_test_scheduling
    onecycle_scheduling = onecycle_scheduling


def callbacks(hparams: Namespace) -> List[Callback]:
    logdir = ResNetArgs.info_from_args(hparams, info="logpath")
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
    logdir = ResNetArgs.info_from_args(hparams, info="logpath")
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
    parser = ResNetArgs.program_level_parser()
    parser = ResNetArgs.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    dm = CovidCTDataModule(hparams)
    model = CovidLightningResNet(hparams)
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
