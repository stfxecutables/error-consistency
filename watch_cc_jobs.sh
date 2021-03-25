#!/bin/bash
if [ -z "$1" ]
then
    echo "usage: ./watch_cc_jobs NODELIST ('NODELIST' available from \`sq\` on Compute Canada)"
else
    echo "Tensorboard running at http://localhost:6008" &&
    echo 'lr-SGD|train_acc_step|train_loss_step|test_acc$|val_loss|val_acc$|^epoch' &&
    echo 'http://localhost:6008/#scalars&regexInput=ResNet&tagFilter=lr-SGD%7Ctrain_acc_step%7Ctrain_loss_step%7Ctest_acc%24%7Cval_loss%7Cval_acc%24%7C%5Eepoch&_smoothingWeight=0.75' &&
    ssh -N -i ~/.ssh/id_siku -L localhost:6008:$1:6006 dberger@siku.ace-net.ca
fi
