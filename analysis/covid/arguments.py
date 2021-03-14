from argparse import ArgumentParser, Namespace
from pathlib import Path


class EfficientNetArgs:
    @staticmethod
    def program_level_parser() -> ArgumentParser:
        parser = ArgumentParser()
        # program args
        parser.add_argument(
            "--version", type=str, choices=[f"b{i}" for i in range(8)], default="b0"
        )
        parser.add_argument("--batch-size", type=int, default=40)
        parser.add_argument("--num-workers", type=int, default=6)
        parser.add_argument("--max-epochs", type=int, default=5000)
        return parser

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        # model-specific args
        lr_choices = ["cosine", "cyclic", "linear-test", "one-cycle", "none", "None"]
        cyclic_choices = ["tr", "triangular", "triangular2", "tr2", "gamma", "exp_range"]
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--pretrain", action="store_true")  # i.e. do pre-train if flag
        parser.add_argument("--initial-lr", type=float, default=0.001)
        parser.add_argument("--weight-decay", type=float, default=0.00001)

        # learning rate args
        parser.add_argument("--lr-schedule", choices=lr_choices)
        parser.add_argument("--onecycle-pct", type=float, default=0.05)
        parser.add_argument("--lrtest-min", type=float, default=1e-6)
        parser.add_argument("--lrtest-max", type=float, default=0.05)
        parser.add_argument("--lrtest-epochs-to-max", type=float, default=1500)
        parser.add_argument("--cyclic-mode", choices=cyclic_choices, default="gamma")
        parser.add_argument("--cyclic-max", type=float, default=0.01)
        parser.add_argument("--cyclic-base", type=float, default=1e-4)

        # augmentation params
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--elastic-scale", type=float, default=0.1)
        parser.add_argument("--elastic-trans", type=float, default=0.3)
        parser.add_argument("--elastic-shear", type=float, default=0.1)
        parser.add_argument("--elastic-degree", type=float, default=5)
        parser.add_argument("--no-elastic", action="store_true")
        parser.add_argument("--no-rand-crop", action="store_true")
        parser.add_argument("--no-flip", action="store_true")
        parser.add_argument("--noise", action="store_true")
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
