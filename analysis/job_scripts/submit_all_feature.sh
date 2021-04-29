#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=08:00:00
#SBATCH --job-name=feature
#SBATCH --output=feature%A_array%a__%j.out
#SBATCH --array=0-32
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
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn1 --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn3 --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn5 --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn10 --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=lr --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=svm --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=rf --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=ada --dataset=diabetes --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn1 --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn3 --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn5 --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn10 --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=lr --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=svm --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=rf --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=ada --dataset=park --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn1 --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn3 --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn5 --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn10 --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=lr --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=svm --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=rf --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=ada --dataset=trans --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn1 --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn3 --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn5 --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=knn10 --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=lr --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=svm --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=rf --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
"$PYTHON $PROJECT/analysis/feature_selection.py --classifier=ada --dataset=spect --kfold-reps=50 --n-percents=200 --results-dir=/home/derek/Desktop/error-consistency/analysis/results/dfs --cpus=8"
)
eval ${commands["$SLURM_ARRAY_TASK_ID"]}
