#!/bin/bash
module load python/3.7.4
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.4 --weight-decay=5e-3\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.4 --weight-decay=5e-3\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.4 --weight-decay=1e-2\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.4 --weight-decay=1e-2\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.6 --weight-decay=5e-3\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.6 --weight-decay=5e-3\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.6 --weight-decay=1e-2\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
python analysis/covid/generate_jobscripts.py --version=b1 --pretrain\
    --lr-schedule=cyclic --cyclic-max=1e-1 --cyclic-base=1e-4\
    --max-epochs=2000 --batch-size=128\
    --dropout=0.6 --weight-decay=1e-2\
    --elastic-scale=0.3 --elastic-trans=0.3 --elastic-shear=0.2 --elastic-degree=5
