#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=1:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --job-name=tboard
#SBATCH --output=%x__%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# --gres=gpu:v100:1
# --partition=all_gpus
PROJECT=$HOME/projects/def-jlevman/dberger/error-consistency
LOGS=$PROJECT/analysis/covid/logs

module load python/3.8.2
cd $SLUM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
# PYTHON=$(which python)

echo "Starting Tensorboard"
tensorboard --logdir=$LOGS --host 0.0.0.0
