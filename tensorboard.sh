echo "http://localhost:6006/#scalars&tagFilter=train_acc%7Ctrain_loss%7Cval_acc%24%7Cval_loss%7Ctest_acc%24&regexInput=ResNet18"
echo "train_acc|train_loss|val_acc$|val_loss|test_acc$"
tensorboard --logdir analysis/covid/logs --max_reload_threads 4


