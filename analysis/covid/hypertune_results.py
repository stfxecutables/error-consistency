import pandas as pd
import sys
import numpy as np

from pathlib import Path
from time import strftime

from pytorch_lightning import Trainer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analysis.covid.resnet import CovidLightningResNet, trainer_defaults
from analysis.covid.arguments import TUNABLE_PARAMS, to_namespace
from analysis.covid.datamodule import CovidCTDataModule
from analysis.covid.hypertune import PARAMS_DICT


RAY_DIR = Path.home() / "ray_results"
if __name__ == "__main__":
    # CovidLightningResNet.from_ray_id(ray_dir=RAY_DIR, trial_id="4fc8c_00000")
    dfs = []
    for model, config in CovidLightningResNet.from_lightning_logs(num=200):
        try:
            dm = CovidCTDataModule(config)
            defaults = trainer_defaults(config)
            trainer = Trainer.from_argparse_args(to_namespace(config), **defaults)
            test_acc = trainer.test(model, datamodule=dm, verbose=False)[0]["test_acc"]
            dm.setup(stage="fit")
            val_acc = trainer.test(model, test_dataloaders=dm.val_dataloader(), verbose=False)[0][
                "test_acc"
            ]
            tunables = {}
            for param in TUNABLE_PARAMS:
                if param in config:
                    tunables[param] = config[param]
            df = pd.DataFrame(tunables, index=[0]).rename(columns=PARAMS_DICT)
            df.drop(columns="lrtest_epochs_to_max", inplace=True)
            cols = df.columns
            df["test_acc"] = np.round(test_acc, 3)
            df["val_acc"] = np.round(val_acc, 3)
            df = df.loc[:, ["test_acc", "val_acc", *cols]]  # sort so "acc" column first
            dfs.append(df)
        except Exception as e:
            print(e)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by="test_acc", ascending=False, inplace=True)
    print(df)
    timestamp = strftime("%b%d-%H:%M")
    outdir = Path(__file__).resolve().parent / "ray_results"
    outfile = outdir / f"test_results_{timestamp}.json"
    df.to_json(outfile)

