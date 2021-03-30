from warnings import filterwarnings
import os
import pandas as pd
import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import strftime
from typing import Any, Dict, List, no_type_check
from pandas import DataFrame

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.sample import loguniform
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.resnet import CovidLightningResNet, trainer_defaults
from analysis.covid.arguments import CYCLIC_CHOICES, ResNetArgs, get_analysis_params, to_namespace
from analysis.covid.datamodule import CovidCTDataModule


RAY_DIR = Path.home() / "ray_results"
if __name__ == "__main__":
    # CovidLightningResNet.from_ray_id(ray_dir=RAY_DIR, trial_id="4fc8c_00000")
    models, configs = CovidLightningResNet.from_lightning_logs(num=10)
    for model, config in zip(models, configs):
        try:
            dm = CovidCTDataModule(config)
            defaults = trainer_defaults(config)
            trainer = Trainer.from_argparse_args(to_namespace(config), **defaults)
            trainer.test(model, datamodule=dm)
            print("Blorp")
        except Exception as e:
            print(e)

