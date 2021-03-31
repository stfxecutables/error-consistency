#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=0-08:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --signal=INT@300
#SBATCH --job-name=asha_resnet
#SBATCH --output=asha_resnet_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
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

echo "Starting ASHA tuning with ray on $(date)."
tensorboard --logdir=$LOGS --host 0.0.0.0 &
python $PROJECT/analysis/covid/hypertune.py &&\
echo "Finished ASHA tuning with ray on $(date)."
