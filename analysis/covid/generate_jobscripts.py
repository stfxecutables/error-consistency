import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.arguments import EfficientNetArgs


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
#SBATCH --time=0-05:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --job-name=eff{version}{pre}
#SBATCH --output=eff{version}{pre}__%j.out
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


def script_from_args() -> None:
    parser = EfficientNetArgs.program_level_parser()
    parser = EfficientNetArgs.add_model_specific_args(parser)
    args = parser.parse_args()
    pre = "_pre" if args.pretrain else ""
    script_path = SCRIPTS_DIR / EfficientNetArgs.info_from_args(args, info="scriptname")
    version = args.version
    script = TEMPLATE.format(version=version, args=" ".join(sys.argv[1:]), pre=pre)
    with open(script_path, "w") as file:
        file.writelines(script)
    print(f"Saved job submission script to {script_path}")


if __name__ == "__main__":
    script_from_args()
