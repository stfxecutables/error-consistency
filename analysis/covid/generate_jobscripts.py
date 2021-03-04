import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path


SCRIPT = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=1-00:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --job-name=effnet_{version}
#SBATCH --output=eff{version}%A_array%a__%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=all_gpus
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

PROJECT=$HOME/projects/def-jlevman/dberger/error-consistency

module load python/3.8.2
cd $SLUM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
# PYTHON=$(which python)

echo "Starting training at $(date)."
python $PROJECT/analysis/covid/efficientnet.py
echo "Training finished at $(date)."
"""

TIME = "1-00:00:00"
VERSION = "v0"

TEMPLATE = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=1-00:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --job-name=efficientnet_{version}
#SBATCH --output=eff{version}%A_array%a__%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=all_gpus
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

PROJECT=$HOME/projects/def-jlevman/dberger/error-consistency
LOGS=$PROJECT/analysis/covid/logs

module load python/3.8.2
cd $SLUM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
# PYTHON=$(which python)

echo "Starting training efficientnet-{version} with args {args} at $(date)."
tensorboard --logdir=$LOGS --host 0.0.0.0 &
python $PROJECT/analysis/covid/efficientnet.py {args} && \\
echo "Finished training efficientnet-{version} with args {args} at $(date)."
"""

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "job_scripts"
if not SCRIPTS_DIR.exists():
    os.makedirs(SCRIPTS_DIR, exist_ok=True)


def scriptname_from_args(args: Namespace) -> str:
    # model
    hp = args
    ver = hp.version
    pre = "-pretrained" if hp.pretrain else ""

    # learning rate-related
    lr = f"lr0={hp.initial_lr:1.2e}"
    sched = hp.lr_schedule
    if sched == "linear-test":
        sched = str(sched).upper()
    wd = f"L2={hp.weight_decay:1.2e}"
    b = hp.batch_size

    # augments
    crop = "crop" if not hp.no_rand_crop else ""
    flip = "rflip" if not hp.no_flip else ""
    elas = "elstic" if not hp.no_elastic else ""
    noise = "noise" if hp.noise else ""
    augs = f"{crop}+{flip}+{elas}+{noise}".replace("++", "+")
    if augs[-1] == "+":
        augs = augs[:-1]

    scriptname = f"submit__eff-net-{ver}{pre}_{sched}_{lr}_{wd}_{b}_batch_{augs}.sh"
    return scriptname


def script_from_args() -> str:
    parser = ArgumentParser()

    # program args
    parser.add_argument("--version", type=str, choices=[f"b{i}" for i in range(8)], default="b0")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=6)

    # model-specific args
    parser.add_argument("--pretrain", action="store_true")  # i.e. do pre-train if flag
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--lr-schedule", choices=["cosine", "cyclic", "linear-test"])

    # augmentation params
    parser.add_argument("--no-elastic", action="store_true")
    parser.add_argument("--no-rand-crop", action="store_true")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--noise", action="store_true")

    args = parser.parse_args()
    script_path = SCRIPTS_DIR / scriptname_from_args(args)
    version = args.version
    script = TEMPLATE.format(version=version, args=" ".join(sys.argv[1:]))
    with open(script_path, "w") as file:
        file.writelines(script)
    print(f"Saved job submission script to {script_path}")


if __name__ == "__main__":
    script_from_args()
