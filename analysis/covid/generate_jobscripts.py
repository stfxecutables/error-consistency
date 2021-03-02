SCRIPT = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=1-00:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --job-name=efficientnet_v0
#SBATCH --output=effv0%A_array%a__%j.out
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
#SBATCH --time={time}  # [dd]-[hh]:[mm]:[ss]
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

module load python/3.8.2
cd $SLUM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
# PYTHON=$(which python)

echo "Starting training at $(date)."
python $PROJECT/analysis/covid/efficientnet.py
echo "Training finished at $(date)."
"""


if __name__ == "__main__":
