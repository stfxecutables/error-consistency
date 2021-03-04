#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=1-00:00:00  # [dd]-[hh]:[mm]:[ss]
#SBATCH --job-name=effnet
#SBATCH --output=%x__%j.out  # %x is job-name
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=all_gpus
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# program args
# MAKE SURE first arguments are --version b1 with the space for proper print
# "--version", type=str, choices=[f"b{i}" for i in range(8)], default="b0"
# "--batch-size", type=int, default=40
# "--num-workers", type=int, default=6
# "--pretrain", action="store_true"  # i.e. do pre-train if flag
# "--initial-lr", type=float, default=0.001
# "--weight-decay", type=float, default=0.00001
# "--lr-schedule", choices=["cosine", "cyclic", "linear-test"]
# "--no-elastic", action="store_true"
# "--no-rand-crop", action="store_true"
# "--no-flip", action="store_true"
# "--noise", action="store_true"

PROJECT=$HOME/projects/def-jlevman/dberger/error-consistency

module load python/3.8.2
cd $SLUM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
# PYTHON=$(which python)

echo "Starting training efficientnet-$2 with args $@ at $(date)." && \
python $PROJECT/analysis/covid/efficientnet.py && \
echo "Training of efficientnet-$2 with args $@ finished at $(date)."

