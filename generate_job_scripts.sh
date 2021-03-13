module load python/3.7.4
python analysis/covid/generate_jobscripts.py --version=b0 $@
python analysis/covid/generate_jobscripts.py --version=b1 $@
python analysis/covid/generate_jobscripts.py --version=b0 --pretrain $@
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain $@
