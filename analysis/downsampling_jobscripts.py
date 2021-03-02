import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.constants import KFOLD_REPS, N_PERCENTS, RESULTS_DIR

CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
SCRIPT_OUTDIR = Path(__file__).resolve().parent / "job_scripts"
if not SCRIPT_OUTDIR.exists():
    os.makedirs(SCRIPT_OUTDIR, exist_ok=True)


def generate_script(
    time: str = "08:00:00",
    mlp_time: str = "4-00:00:00",
    job_name: str = "downsampling",
    mlp_job_name: str = "downsampling_mlp",
    results_dir: Path = RESULTS_DIR,
    kfold_reps: int = KFOLD_REPS,
    n_percents: int = N_PERCENTS,
    cpus: int = 8,
    script_outdir: Path = SCRIPT_OUTDIR,
) -> Tuple[str, str]:
    lines: List[str]
    mlp_lines: List[str]
    template = '"$PYTHON $PROJECT/analysis/downsampling.py --classifier={classifier} --dataset={dataset} --kfold-reps={kfold_reps} --n-percents={n_percents} --results-dir={results_dir} --cpus={cpus}"'
    lines, mlp_lines = [], []
    for dataset in DATASET_CHOICES:
        for classifier in CLASSIFIER_CHOICES:
            appender = mlp_lines if classifier == "mlp" else lines
            appender.append(
                template.format(
                    classifier=classifier,
                    dataset=dataset,
                    kfold_reps=kfold_reps,
                    n_percents=n_percents,
                    results_dir=results_dir,
                    cpus=cpus,
                )
            )
    N = int(len(lines))
    N_mlp = int(len(mlp_lines))
    bash_array = "\n".join(lines)
    bash_array_mlp = "\n".join(mlp_lines)
    header = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}%A_array%a__%j.out
#SBATCH --array=0-{N}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL"""
    all_header = header.format(time=time, job_name=job_name, N=N)
    mlp_header = header.format(time=mlp_time, job_name=mlp_job_name, N=N_mlp)

    script = """
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
{}
)
eval ${{commands["$SLURM_ARRAY_TASK_ID"]}}
"""
    all_script = f"{all_header}{script.format(bash_array)}"
    mlp_script = f"{mlp_header}{script.format(bash_array_mlp)}"
    with open(script_outdir / "submit_all_downsampling.sh", mode="w") as file:
        file.write(all_script)
    print(f"Saved downsampling script to {script_outdir / 'submit_all_downsampling.sh'}")
    with open(script_outdir / "submit_mlp_downsampling.sh", mode="w") as file:
        file.write(mlp_script)
    print(f"Saved mlp downsampling script to {script_outdir / 'submit_mlp_downsampling.sh'}")

    return all_script, mlp_script


if __name__ == "__main__":
    print(f"Will save scripts in {SCRIPT_OUTDIR}")
    scripts = generate_script(kfold_reps=50, n_percents=200, cpus=8)
    print(scripts[0])
    print(scripts[1])
