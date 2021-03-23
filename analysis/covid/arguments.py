import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

IN_COMPUTE_CANADA_JOB = os.environ.get("SLURM_TMPDIR") is not None
ON_COMPUTE_CANADA = os.environ.get("CC_CLUSTER") is not None
# https://pytorch.org/hub/pytorch_vision_resnet/ default=18,
LR_CHOICES = ["cosine", "cyclic", "linear-test", "one-cycle", "none", "None"]
CYCLIC_CHOICES = ["tr", "triangular", "triangular2", "tr2", "gamma", "exp_range"]
EFFNET_CHOICES = [f"b{i}" for i in range(8)]
RESNET_CHOICES = [18, 34, 50, 101, 152]

PROG_ARGS: Dict[str, Dict] = {
    "--batch-size": dict(type=int, default=40),
    "--num-workers": dict(type=int, default=6),
    "--max-epochs": dict(type=int, default=1000),
}

MODEL_ARGS: Dict[str, Dict] = {
    "--pretrain": dict(action="store_true"),  # i.e. do pre-train if flag
    "--initial-lr": dict(type=float, default=0.001),
    "--weight-decay": dict(type=float, default=0.00001),
}

LR_ARGS: Dict[str, Dict] = {
    "--lr-schedule": dict(choices=LR_CHOICES),
    "--onecycle-pct": dict(type=float, default=0.05),
    "--lrtest-min": dict(type=float, default=1e-6),
    "--lrtest-max": dict(type=float, default=0.05),
    "--lrtest-epochs-to-max": dict(type=float, default=1500),
    "--cyclic-mode": dict(choices=CYCLIC_CHOICES, default="gamma"),
    "--cyclic-max": dict(type=float, default=0.01),
    "--cyclic-base": dict(type=float, default=1e-4),
    "--cyclic-f": dict(type=int, default=60),
}

AUG_ARGS: Dict[str, Dict] = {
    "--dropout": dict(type=float, default=0.2),
    "--elastic-scale": dict(type=float, default=0.1),
    "--elastic-trans": dict(type=float, default=0.3),
    "--elastic-shear": dict(type=float, default=0.1),
    "--elastic-degree": dict(type=float, default=5),
    "--no-elastic": dict(action="store_true"),
    "--no-rand-crop": dict(action="store_true"),
    "--no-flip": dict(action="store_true"),
    "--noise": dict(action="store_true"),
}

TUNABLE_PARAMS = (
    ["--batch-size"] + list(MODEL_ARGS.keys()) + list(LR_ARGS.keys()) + list(AUG_ARGS.keys())
)
TUNABLE_PARAMS = list(map(lambda s: s.replace("--", "").replace("-", "_"), TUNABLE_PARAMS))


def get_log_params(config: Dict[str, Any]) -> List[str]:
    tunable = ["version"]
    tunable.extend(list(PROG_ARGS.keys()))
    tunable.extend(list(MODEL_ARGS.keys()))
    lr_schedule = config["lr_schedule"]
    if lr_schedule in ["None", "none", None]:
        pass
    elif lr_schedule == "cyclic":
        args = list(LR_ARGS.keys())
        tunable.extend(list(filter(lambda s: "cyclic" in s, args)))
    elif lr_schedule == "linear-test":
        args = list(LR_ARGS.keys())
        tunable.extend(list(filter(lambda s: "lrtest" in s, args)))
    elif lr_schedule == "one-cycle":
        tunable.extend(["onecycle-pct"])
    else:
        raise ValueError("Unknown / unimplemented lr_schedule.")
    tunable.extend(list(AUG_ARGS.keys()))
    tunable = list(map(lambda s: s.replace("--", "").replace("-", "_"), tunable))
    return tunable


def get_analysis_params() -> List[str]:
    config_tunable = [f"config.{param}" for param in TUNABLE_PARAMS]
    config_tunable = list(filter(lambda p: "lrtest" not in p, config_tunable))
    return [
        "date",
        "val_acc",
        "val_loss",
        "epoch",
        "training_iteration",
        *config_tunable,
        "time_total_s",
    ]


class EfficientNetArgs:
    @staticmethod
    def program_level_parser() -> ArgumentParser:
        parser = ArgumentParser()
        # program args
        parser.add_argument("--version", type=str, choices=EFFNET_CHOICES, default="b0")
        for argname, kwargs in PROG_ARGS.items():
            parser.add_argument(argname, **kwargs)
        return parser

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        # model-specific args
        parser = ArgumentParser(parents=[parser], add_help=False)
        for argname, kwargs in MODEL_ARGS.items():
            parser.add_argument(argname, **kwargs)
        for argname, kwargs in LR_ARGS.items():
            parser.add_argument(argname, **kwargs)
        for argname, kwargs in AUG_ARGS.items():
            parser.add_argument(argname, **kwargs)
        return parser

    @staticmethod
    def defaults() -> Namespace:
        parser = EfficientNetArgs.program_level_parser()
        parser = EfficientNetArgs.add_model_specific_args(parser)
        return parser.parse_args()

    @staticmethod
    def info_from_args(args: Namespace, info: str) -> str:
        hp = args
        ver = hp.version
        pre = "-pretrained" if hp.pretrain else ""

        # learning rate-related
        sched = hp.lr_schedule
        lr = f"lr0={hp.initial_lr:1.2e}"
        is_range_test = sched == "linear-test"
        if sched == "one-cycle":
            sched = f"{sched}{hp.onecycle_pct:1.2f}"
        elif is_range_test:
            sched = str(sched).upper()
            lr = f"lr-max={hp.lrtest_max}@{hp.lrtest_epochs_to_max}"
        elif sched == "cyclic":
            mode = hp.cyclic_mode
            if mode in ["tr", "triangular"]:
                m = "tr"
            elif mode in ["tr2", "triangular2"]:
                m = "tr2"
            else:
                m = "exp"
            lr_max = hp.cyclic_max
            lr_base = hp.cyclic_base
            f = hp.cyclic_f
            if mode in ["gamma", "exp_range"]:
                lr = f"cyc-{m}=({lr_base:1.1e},{lr_max:1.1e},f={f})"
            else:
                lr = f"cyc-{m}=({lr_base:1.1e},{lr_max:1.1e})"
        wd = f"L2={hp.weight_decay:1.2e}"
        b = hp.batch_size
        e = hp.max_epochs

        # augments
        crop = "crop" if not hp.no_rand_crop else ""
        flip = "rflip" if not hp.no_flip else ""
        elas = "elst" if not hp.no_elastic else ""
        if elas != "":
            elas = (
                f"{elas}(sc={hp.elastic_scale}_tr={hp.elastic_trans}"
                f"_sh={hp.elastic_shear}_deg={hp.elastic_degree})"
            )
        noise = "noise" if hp.noise else ""
        drop = f"drp{hp.dropout:0.1f}"
        augs = f"{crop}+{flip}+{elas}+{noise}+{drop}".replace("++", "+")
        if augs[-1] == "+":
            augs = augs[:-1]
        if augs[0] == "+":
            augs = augs[1:]

        if info == "scriptname":
            return f"submit__eff-net-{ver}{pre}_{sched}_{lr}_{wd}_{b}batch_{e}ep_{augs}.sh"
        elif info == "logpath":
            version_dir = Path(__file__).resolve().parent / f"logs/efficientnet-{ver}{pre}/{sched}"
            dirname = f"{lr}_{wd}_{b}batch_{e}ep_{augs}"
            return str(version_dir / dirname)
        else:
            raise ValueError("`info` param must be one of 'scriptname' or 'logpath'.")


class ResNetArgs:
    @staticmethod
    def program_level_parser() -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--version", type=int, choices=RESNET_CHOICES, default=18)
        parser.add_argument("--resnet", type=bool, default=True)
        for argname, kwargs in PROG_ARGS.items():
            parser.add_argument(argname, **kwargs)
        return parser

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parser], add_help=False)
        for argname, kwargs in MODEL_ARGS.items():
            parser.add_argument(argname, **kwargs)
        for argname, kwargs in LR_ARGS.items():
            parser.add_argument(argname, **kwargs)
        for argname, kwargs in AUG_ARGS.items():
            parser.add_argument(argname, **kwargs)
        return parser

    @staticmethod
    def defaults() -> Namespace:
        parser = ResNetArgs.program_level_parser()
        parser = ResNetArgs.add_model_specific_args(parser)
        return parser.parse_args()

    @staticmethod
    def info_from_args(args: Union[Namespace, Dict[str, Any]], info: str) -> str:
        if isinstance(args, dict):

            class Dummy:
                def __init__(self, args: Dict[str, Any]) -> None:
                    for key, val in args.items():
                        setattr(self, key, val)

            hp: Any = Dummy(args)
        else:
            hp = args
        ver = hp.version
        pre = "-pretrained" if hp.pretrain else ""

        # learning rate-related
        sched = hp.lr_schedule
        lr = f"lr0={hp.initial_lr:1.2e}"
        is_range_test = sched == "linear-test"
        if sched == "one-cycle":
            sched = f"{sched}{hp.onecycle_pct:1.2f}"
        elif is_range_test:
            sched = str(sched).upper()
            lr = f"lr-max={hp.lrtest_max}@{hp.lrtest_epochs_to_max}"
        elif sched == "cyclic":
            mode = hp.cyclic_mode
            if mode in ["tr", "triangular"]:
                m = "tr"
            elif mode in ["tr2", "triangular2"]:
                m = "tr2"
            else:
                m = "exp"
            lr_max = hp.cyclic_max
            lr_base = hp.cyclic_base
            f = hp.cyclic_f
            if mode in ["gamma", "exp_range"]:
                lr = f"cyc-{m}=({lr_base:1.1e},{lr_max:1.1e},f={f})"
            else:
                lr = f"cyc-{m}=({lr_base:1.1e},{lr_max:1.1e})"
        wd = f"L2={hp.weight_decay:1.2e}"
        b = hp.batch_size
        e = hp.max_epochs

        # augments
        crop = "crop" if not hp.no_rand_crop else ""
        flip = "rflip" if not hp.no_flip else ""
        elas = "elst" if not hp.no_elastic else ""
        if elas != "":
            elas = (
                f"{elas}(sc={hp.elastic_scale}_tr={hp.elastic_trans}"
                f"_sh={hp.elastic_shear}_deg={hp.elastic_degree})"
            )
        noise = "noise" if hp.noise else ""
        drop = f"drp{hp.dropout:0.1f}"
        augs = f"{crop}+{flip}+{elas}+{noise}+{drop}".replace("++", "+")
        if augs[-1] == "+":
            augs = augs[:-1]
        if augs[0] == "+":
            augs = augs[1:]

        if info == "scriptname":
            return f"submit__ResNet{ver}{pre}_{sched}_{lr}_{wd}_{b}batch_{e}ep_{augs}.sh"
        elif info == "logpath":
            version_dir = Path(__file__).resolve().parent / f"logs/ResNet{ver}{pre}/{sched}"
            dirname = f"{lr}_{wd}_{b}batch_{e}ep_{augs}"
            return str(version_dir / dirname)
        else:
            raise ValueError("`info` param must be one of 'scriptname' or 'logpath'.")
