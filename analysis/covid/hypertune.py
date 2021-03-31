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
from analysis.covid.resnet import CovidLightningResNet
from analysis.covid.arguments import CYCLIC_CHOICES, ResNetArgs, get_analysis_params, to_namespace
from analysis.covid.datamodule import CovidCTDataModule

TIMESTAMP = strftime("%b%d-%Y-%H:%M")
RAY_RESULTS = Path(__file__).resolve().parent / "ray_results"
RAY_RESULTS_ALL = RAY_RESULTS / "all_results_df.json"
RAY_RESULTS_CURRENT = RAY_RESULTS / f"results_{TIMESTAMP}.json"
if not RAY_RESULTS.exists():
    os.makedirs(RAY_RESULTS, exist_ok=True)

RESNET_VERSION = 34
IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None
GPU_CCANADA = 1 / 8 if RESNET_VERSION == 18 else 1 / 6
GPU = GPU_CCANADA if ON_COMPUTE_CANADA else 1 / 3
NUM_SAMPLES = 128 if ON_COMPUTE_CANADA else 6
GRACE_PERIOD = 25
MAX_T = 151


BASE_BATCHES = [4, 8, 16, 32, 64] if ON_COMPUTE_CANADA else [4, 8, 16, 32]
BASE_LR_MIN, BASE_LR_MAX = 1e-8, 1e-4
BASE_LR_BOUNDARY = tune.loguniform(BASE_LR_MIN, BASE_LR_MAX).sample()
BASE_CYCLIC_CHOICES = filter(lambda s: s in ["gamma", "exp_range"], CYCLIC_CHOICES)
BASE_CONFIG = dict(
    # basic / model
    # pretrain=tune.choice([True, False]),
    version=RESNET_VERSION,
    pretrain=tune.choice([True]),
    batch_size=tune.choice(BASE_BATCHES),
    output=tune.choice(["gap", "linear"]),
    # learning rates
    initial_lr=tune.loguniform(BASE_LR_MIN, BASE_LR_MAX),
    lr_min=tune.loguniform(BASE_LR_MIN, BASE_LR_BOUNDARY),
    lr_max=tune.loguniform(BASE_LR_BOUNDARY + BASE_LR_MIN, BASE_LR_MAX),
    lr_schedule=tune.choice(["step", "exp", "None"]),  # None is Adam
    cyclic_mode=tune.choice(BASE_CYCLIC_CHOICES),
    cyclic_f=tune.qrandint(10, 100, 10),
    step_size=tune.qrandint(20, 100, 5),
    gamma=tune.quniform(0.1, 0.9, 0.1),
    gamma_sub=tune.uniform(1e-3, 1e-2),
    # regularization
    weight_decay=tune.loguniform(1e-9, 1.0),
    dropout=tune.quniform(0.05, 0.95, 0.05),
    # augments
    no_rand_crop=False,
    noise=True,
    no_flip=True,
    no_elastic=False,
    elastic_scale=tune.uniform(1e-3, 0.4),
    elastic_trans=tune.uniform(1e-3, 0.4),
    elastic_shear=tune.uniform(1e-3, 0.3),
    elastic_degree=tune.randint(1e-3, 10),
    noise_sd=tune.uniform(1e-3, 1.0),
)

TWEAK_BATCHES = [4, 8, 16, 32, 64]
TWEAK_LR_MIN, TWEAK_LR_MAX = 1e-6, 1e-2
TWEAK_LR_BOUNDARY = tune.loguniform(TWEAK_LR_MIN, TWEAK_LR_MAX).sample()
TWEAK_CYCLIC_CHOICES = BASE_CYCLIC_CHOICES
TWEAK_SCHEDULES = [None]
TWEAK_CONFIG = dict(
    # basic / model
    pretrain=tune.choice([True, False]),
    batch_size=tune.choice(TWEAK_BATCHES),
    output=tune.choice(["gap", "linear"]),
    # learning rates
    initial_lr=tune.loguniform(TWEAK_LR_MIN, TWEAK_LR_MAX),
    lr_min=tune.loguniform(TWEAK_LR_MIN, TWEAK_LR_BOUNDARY),
    lr_max=tune.loguniform(TWEAK_LR_BOUNDARY + TWEAK_LR_MIN * 5, TWEAK_LR_MAX),
    lr_schedule=tune.choice(TWEAK_SCHEDULES),
    cyclic_mode=tune.choice(TWEAK_CYCLIC_CHOICES),
    cyclic_f=tune.qrandint(10, 100, 10),
    step_size=tune.qrandint(20, 100, 5),
    gamma=tune.quniform(0.1, 0.9, 0.1),
    gamma_sub=tune.uniform(1e-3, 1e-2),
    # regularization
    weight_decay=tune.loguniform(1e-6, 1),
    dropout=tune.quniform(0.65, 0.95, 0.05),
    # augments
    no_rand_crop=False,
    noise=True,
    no_flip=True,
    no_elastic=False,
    elastic_scale=tune.uniform(1e-3, 0.4),
    elastic_trans=tune.uniform(1e-3, 0.4),
    elastic_shear=tune.uniform(1e-3, 0.3),
    elastic_degree=tune.randint(1e-3, 10),
    noise_sd=tune.uniform(1e-3, 1.0),
)

# Short names solely for CLI Reporter readability
PARAMS_DICT = dict(
    # model
    version="v",
    batch_size="btch",
    pretrain="pre",
    output="out",
    weight_decay="L2",
    # learning rates
    initial_lr="lr",
    lr_schedule="sched",
    lr_min="lr-min",
    lr_max="lr-max",
    cyclic_mode="cyc-mode",
    cyclic_f="cyc-f",
    step_size="lr-step",
    gamma="step-g",
    gamma_sub="g-sub",
    # augmentation / regularization
    dropout="drop",
    elastic_scale="e-scale",
    elastic_trans="e-transl",
    elastic_shear="e-shear",
    elastic_degree="e-deg",
    no_flip="no-flp",
    noise_sd="noise-sd",
)
METRICS_DICT = dict(val_loss="loss", val_acc="acc", val_epoch="epoch", training_iteration="step")

COLUMNS_FMT = dict(
    val_acc="0.3f",
    val_loss="1.5f",
    steps="",
    batch="",
    pre="",
    output="",
    lr0="1.2e",
    L2="1.2e",
    lr_sched="",
    lr_min="1.2e",
    lr_max="1.2e",
    cyc_mode="",
    cyc_f="",
    step_size="",
    gamma="0.1f",
    gamma_sub="1.2e",
    dropout="0.2f",
    el_scale="0.3f",
    el_trans="0.3f",
    el_shear="0.3f",
    el_degree="",
    no_el="",
    no_rand_crop="",
    no_flip="",
    noise="",
    noise_sd="0.2f",
    time_total_s="0.0f",
)


def trainer_defaults(config: Dict[str, Any]) -> Dict:
    logdir = ResNetArgs.info_from_args(config, info="logpath")
    refresh_rate = 0 if IN_COMPUTE_CANADA_JOB else None
    max_epochs = 800 if config["lr_schedule"] == "linear-test" else config["max_epochs"]
    config["max_epochs"] = max_epochs
    return dict(
        default_root_dir=logdir,
        progress_bar_refresh_rate=refresh_rate,
        gpus=1,
        max_epochs=max_epochs,
        min_epochs=10,
    )


def callbacks(config: Dict[str, Any]) -> List[Callback]:
    logdir = ResNetArgs.info_from_args(config, info="logpath")
    cbs = [
        LearningRateMonitor(logging_interval="epoch") if config["lr_schedule"] else None,
        EarlyStopping("val_acc", min_delta=0.001, patience=30, mode="max"),
        ModelCheckpoint(
            dirpath=logdir,
            filename="{epoch}-{step}_{val_acc:.2f}_{train_acc:0.3f}",
            monitor="val_acc",
            save_last=True,
            save_top_k=3,
            mode="max",
            save_weights_only=False,
        ),
        TuneReportCallback(["val_acc", "val_loss", "val_epoch"], on="validation_end"),
    ]
    return list(filter(lambda c: c is not None, cbs))  # type: ignore


# def train_resnet(config: Dict[str, Any], num_workers=1, num_epochs=2):
#     Operator = TrainingOperator.from_ptl(CovidLightningResNet)
#     trainer = TorchTrainer(
#         training_operator_cls=Operator, num_workers=num_workers, use_tqdm=False, use_gpu=True
#     )
#     for _ in range(num_epochs):
#         trainer.train()
#     trainer.shutdown()


def tune_resnet(config: Dict[str, Any]) -> None:
    filterwarnings("ignore", message=".*The dataloader.*", category=UserWarning)
    filterwarnings("ignore", message=".*To copy construct.*", category=UserWarning)
    filterwarnings(
        "ignore", message=".*You are using `LearningRateMonitor`.*", category=RuntimeWarning
    )
    defaults = {**trainer_defaults(config), **dict(progress_bar_refresh_rate=0)}
    model = CovidLightningResNet(config)
    trainer = Trainer.from_argparse_args(
        to_namespace(config), callbacks=callbacks(config), **defaults
    )
    trainer.fit(model)


# @ray.remote(num_gpus=GPU, max_calls=1)
# NOTE: maybe can use this later to get test results in parallel
@no_type_check
@ray.remote(max_calls=1)
def test_acc(config: Namespace) -> Dict:
    dm = CovidCTDataModule(hparams=config)
    model = CovidLightningResNet(hparams=config)
    # if you read the source, the **kwargs in Trainer.from_argparse_args just call .update on an
    # args dictionary, so you can override what you want with it
    defaults = {**trainer_defaults(config), **dict(progress_bar_refresh_rate=0)}
    trainer = Trainer.from_argparse_args(config, callbacks=callbacks(config), **defaults)
    # trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm)
    return results[0]  # type: ignore


# @ray.remote
def get_config() -> Dict:
    parser = ResNetArgs.program_level_parser()
    parser = ResNetArgs.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args("")
    hparams.num_workers = 2
    return hparams.__dict__  # type: ignore


def process_cmdline_options() -> None:
    global MAX_T
    global GRACE_PERIOD
    parser = ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    if args.download:  # pre-download models for Compute Canada
        for model in [resnet18, resnet34, resnet50, resnet101, resnet152]:
            _ = model(pretrained=True)
            _ = model(pretrained=False)
        sys.exit()
    if args.fast_dev_run:  # gotta go fast
        GRACE_PERIOD = 5
        MAX_T = 10


# Rapidly hits 80% test acc in like 20 epochs
# python analysis/covid/resnet.py --batch-size=32 --version=18\
#       --lr-schedule=None --initial-lr=1.4e-4 --lr-max=0.1 --lr-min=1e-5\
#       --max-epochs=50 --weight-decay=0.006 --noise --noise-sd=0.5\
#       --output=gap --dropout=0.8 --pretrain
if __name__ == "__main__":
    ###########################################################################
    # START RESOLVING ANNOYING COMPUTE CANADA ISSUES:
    process_cmdline_options()
    # sys.path modifications happen only in main thread, need to manually ensure
    # children also inherit / get this. See link below for details.
    # https://stackoverflow.com/questions/54338013/parallel-import-a-python-file-from-sibling-folder
    parent_dir = str(Path(__file__).resolve().parent.parent.parent)
    os.environ["PYTHONPATH"] = f"{parent_dir}:{os.environ.get('PYTHONPATH', '')}"

    # another hack, see https://github.com/ray-project/ray/issues/10995#issuecomment-698177711
    os.environ["SLURM_JOB_NAME"] = "bash"
    # END RESOLVING ANNOYING COMPUTE CANADA ISSUES.
    ###########################################################################

    config = get_config()
    ray.init()
    # fmt: off
    config = {
        **config,
        **BASE_CONFIG,
        # **TWEAK_CONFIG,
    }
    # fmt: on
    scheduler = AsyncHyperBandScheduler(
        time_attr="val_epoch",  # defined in `validation_step`
        grace_period=GRACE_PERIOD,
        max_t=MAX_T,
    )

    reporter = CLIReporter(
        # parameter_columns=get_log_params(),
        parameter_columns=PARAMS_DICT,
        # metric_columns=["val_acc", "val_epoch", "training_iteration"],
        metric_columns=METRICS_DICT,
        max_error_rows=3,
        max_report_frequency=15,
    )

    # avoid shadowing "analysis" folder
    analysis_results = tune.run(
        tune.with_parameters(tune_resnet),
        # tune.with_parameters(train_resnet, num_epochs=10),
        resources_per_trial={"cpu": 2, "gpu": GPU},
        name=f"asha_test__{TIMESTAMP}",
        metric="val_acc",
        mode="max",
        scheduler=scheduler,
        progress_reporter=reporter,
        # stop={"training_iteration": 2},
        search_alg=None,  # use random search
        num_samples=NUM_SAMPLES,
        verbose=1,
        config=config,
        # resume=True,
    )
    results: DataFrame = analysis_results.results_df
    columns = list(filter(lambda col: "config." not in col, results.columns))
    results = (
        results.filter(items=get_analysis_params())
        .rename(columns=lambda s: s.replace("config.", ""))
        .rename(columns=lambda s: s.replace("elastic", "el"))
        .rename(columns=lambda s: s.replace("cyclic", "cyc"))
        .rename(
            columns={
                "lr_schedule": "lr_sched",
                "training_iteration": "steps",
                "pretrain": "pre",
                "initial_lr": "lr0",
                "weight_decay": "L2",
                "batch_size": "batch",
            }
        )
    )
    ray.shutdown()

    pd.set_option("display.width", 999)
    pd.set_option("display.max_columns", 999)
    pd.set_option("display.max_rows", 999)
    print(results)
    results.sort_values(by="val_acc", ascending=False, inplace=True)
    results.to_json(RAY_RESULTS_CURRENT)
    print(f"Saved results for current ASHA run to {RAY_RESULTS_CURRENT}")

    if RAY_RESULTS_ALL.exists():
        print("Found previous Ray results. Updating...")
        prev_results = pd.read_json(RAY_RESULTS_ALL)
        results = pd.concat([prev_results, results])
        results.sort_values(by="val_acc", ascending=False, inplace=True)
    else:
        print("No previous Ray results found. Saving current results...")
    results.to_json(RAY_RESULTS_ALL)
    print(f"Saved all results to {RAY_RESULTS_ALL}")

    printable = results.drop(columns=["onecycle_pct", "date"])
    table = printable.to_markdown(
        tablefmt="plain", index=True, floatfmt=[""] + list(COLUMNS_FMT.values())
    )
    print("Summary of all results so far:")
    print(table)

