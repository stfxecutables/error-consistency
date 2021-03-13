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
        parser = ArgumentParser(parents=[parser], add_help=False)
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
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--no-elastic", action="store_true")
        parser.add_argument("--no-rand-crop", action="store_true")
        parser.add_argument("--no-flip", action="store_true")
        parser.add_argument("--noise", action="store_true")
        return parser

    @staticmethod
    def info_from_args(args: Namespace, info: str) -> str:
        hp = args
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
