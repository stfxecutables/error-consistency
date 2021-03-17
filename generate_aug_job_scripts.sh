#!/bin/bash
module load python/3.7.4
# no pre-training, vary L2, cyclic tr2
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-5\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-4\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-3\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-2\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
# same with Adam
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=None\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-5\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=None\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-4\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=None\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-3\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=None\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-2\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
# pre-training, vary L2, cyclic tr2
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-5\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-4\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-3\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-mode=tr2 --cyclic-base=1e-4 --cyclic-max=1e-2\
    --max-epochs=2000 --batch-size=128\
    --weight-decay=1e-2\
    --elastic-scale=0.2 --elastic-trans=0.3 --elastic-shear=0.1 --elastic-degree=5