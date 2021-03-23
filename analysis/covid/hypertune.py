import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple, no_type_check
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.sample import loguniform
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd import TorchTrainer
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.resnet import CovidLightningResNet
from analysis.covid.arguments import ResNetArgs
from analysis.covid.datamodule import CovidCTDataModule

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None


def trainer_defaults(hparams: Namespace) -> Dict:
    logdir = ResNetArgs.info_from_args(hparams, info="logpath")
    refresh_rate = 0 if IN_COMPUTE_CANADA_JOB else None
    max_epochs = 2000 if hparams["lr_schedule"] == "linear-test" else hparams["max_epochs"]
    return dict(
        default_root_dir=logdir,
        progress_bar_refresh_rate=refresh_rate,
        gpus=1,
        max_epochs=max_epochs,
        min_epochs=1000,
    )


def callbacks(config: Dict[str, Any]) -> List[Callback]:
    logdir = ResNetArgs.info_from_args(config, info="logpath")
    return [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping("val_acc", min_delta=0.001, patience=30, mode="max"),
        ModelCheckpoint(
            dirpath=logdir,
            filename="{epoch}-{step}_{val_acc:.2f}_{train_acc:0.3f}",
            monitor="val_acc",
            save_last=True,
            save_top_k=1,
            mode="max",
            save_weights_only=False,
        ),
        # TuneReportCallback({"loss": "pts/val_loss", "acc": "ptl/val_acc"}, on="validation_end"),
        TuneReportCallback({"loss": "val_loss", "acc": "val_acc"}, on="validation_end"),
    ]


def train_resnet(config: Dict[str, Any], num_workers=2, num_epochs=10):
    Operator = TrainingOperator.from_ptl(CovidLightningResNet)
    trainer = TorchTrainer(
        training_operator_cls=Operator, num_workers=num_workers, use_tqdm=False, use_gpu=True
    )
    for _ in range(num_epochs):
        trainer.train()
    trainer.shutdown()


def tune_resnet(config: Dict[str, Any], num_epochs=10):
    model = CovidLightningResNet(config)
    defaults = {**trainer_defaults(hparams=config), **dict(progress_bar_refresh_rate=0)}
    trainer = Trainer.from_argparse_args(
        to_namespace(config), callbacks=callbacks(config), **defaults
    )
    trainer.fit(model)


# @ray.remote(num_gpus=0.333, max_calls=1)
@ray.remote(max_calls=1)
def test_acc(config: Namespace) -> Dict:
    dm = CovidCTDataModule(hparams=config)
    model = CovidLightningResNet(hparams=config)
    # if you read the source, the **kwargs in Trainer.from_argparse_args just call .update on an
    # args dictionary, so you can override what you want with it
    defaults = {**trainer_defaults(hparams=config), **dict(progress_bar_refresh_rate=0)}
    trainer = Trainer.from_argparse_args(config, callbacks=callbacks(config), **defaults)
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm)
    return results[0]  # type: ignore


# @ray.remote
def get_config() -> Dict:
    parser = ResNetArgs.program_level_parser()
    parser = ResNetArgs.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args("")
    hparams.batch_size = 32
    hparams.max_epochs = 2
    hparams.num_workers = 1
    return hparams.__dict__  # type: ignore


def to_namespace(config: Dict[str, Any]) -> Namespace:
    ns: Namespace = Namespace.__new__(Namespace)
    for key, val in config.items():
        setattr(ns, key, val)
    return ns


if __name__ == "__main__":
    # ray.init(num_cpus=6, num_gpus=1)
    config = get_config()
    ray.init()
    config = {**config, **dict(weight_decay=tune.loguniform(1e-6, 1e-1))}
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",  # defined in `validation_step`
        grace_period=20,
        metric="val_acc",
        mode="max",
        max_t=301.0,
    )
    reporter = CLIReporter(
        parameter_columns=["weight_decay"], metric_columns=["loss", "acc", "training_iteration"]
    )

    analysis = tune.run(
        tune.with_parameters(tune_resnet, num_epochs=10),
        resources_per_trial={"cpu": 1, "gpu": 0.333},
        name="asha_test",
        scheduler=scheduler,
        progress_reporter=reporter,
        # stop={"training_iteration": 2},
        num_samples=3,
        verbose=1,
        config=config,
    )
    print(analysis.best_config)
    ray.shutdown()
