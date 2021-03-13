#!/bin/bash
if [ -z "$1" ]
then
    echo "usage: ./watch_cc_jobs NODELIST ('NODELIST' available from \`sq\` on Compute Canada)"
else
    echo "Tensorboard running at http://localhost:6008" &&
    echo 'filter: val_acc$|train_acc|val_loss|test_acc$|lr-Adam|lr-SGD' &&
    ssh -N -i ~/.ssh/id_siku -L localhost:6008:$1:6006 dberger@siku.ace-net.ca
fi
