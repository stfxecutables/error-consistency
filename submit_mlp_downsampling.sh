#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=4-00:00:00
#SBATCH --job-name=downsampling_mlp
#SBATCH --output=downsampling_mlp%A_array%a__%j.out
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
PROJECT=$HOME/projects/def-jlevman/dberger/error-consistency

module load python/3.8.2
cd $SLURM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
PYTHON=$(which python)

# virtualenv --no-download .venv
# source .venv/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r $PROJECT/requirements.txt

commands=(
"$PYTHON $PROJECT/analysis/downsampling.py --classifier=mlp --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/downsampling.py --classifier=mlp --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/downsampling.py --classifier=mlp --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/downsampling.py --classifier=mlp --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
)
eval ${commands["$SLURM_ARRAY_TASK_ID"]}
