#!/bin/bash
if [ -z "$1" ]
then
    echo "usage: ./watch_cc_jobs NODELIST ('NODELIST' available from \`sq\` on Compute Canada)"
else
    echo "Tensorboard running at http://localhost:6008" &&
    echo 'val_acc$|train_acc|val_loss|test_acc$|lr-SGD|epoch|train_loss' &&
    echo 'http://localhost:6008/#scalars&tagFilter=val_acc%24%7Ctrain_acc%7Cval_loss%7Ctest_acc%24%7Clr-SGD%7Cepoch%7Ctrain_acc&regexInput=drp0.6&_smoothingWeight=0.901' &&
    ssh -N -i ~/.ssh/id_siku -L localhost:6008:$1:6006 dberger@siku.ace-net.ca
fi
