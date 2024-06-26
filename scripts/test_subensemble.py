# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
# lightning Changelog: https://lightning.ai/docs/pytorch/stable/upgrade/from_1_7.html
# tutorial: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
import os.path
import torch
import numpy as np
from torchvision import transforms as torch_tfs
from torch.utils.data import DataLoader, Subset
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from alg.resnet_ae import ResnetAutoencoder
from alg.subensemble import SubEnsemble
from alg.subensemble_dataset import SubensembleDataset

def default_arguments(basedir : str):
    subens_file = os.path.join(basedir, "subensemble_files.txt")
    with open(subens_file, 'r') as f:
        lines= [line.rstrip() for line in f]
    
    args = {
        "mdl_pths" : lines,
        "num_classes": 1,
        "resnet" : 18,
        "bs" : 32,
        "dataset" : "data/combined"     # /test        
    }
    return args

def test_subensemble(mdl_pths : list, dataset_path : str, model_settings : dict, logdir : str,
                          img_folder : str = "images", label_folder : str = "labels",
                          subset_n : int = None) -> pd.DataFrame:
    subens_model =   SubEnsemble(
        model_settings["num_classes"],
        18,
        transfer=True,
        heads=mdl_pths[1:],
        load_true = True
    )
    print("Loading autoencoder from: {}".format(mdl_pths[0]))
    resn_ae = ResnetAutoencoder.load_from_checkpoint(checkpoint_path = mdl_pths[0])
    missing_keys, unexp_keys = subens_model.from_AE(resn_ae)
    if missing_keys or unexp_keys: print("Missing Layers: {},\n\nUnexpected Layers: {}".format(missing_keys, unexp_keys))
    subens_model.freeze()

    # 3. load the inference dataset
    mean = (0.5,)
    std = (0.5,)
    tfs = torch_tfs.Compose([
        torch_tfs.ConvertImageDtype(torch.float),
        torch_tfs.Normalize(mean, std)        
    ])
    base_ds = SubensembleDataset(
        root=dataset_path,
        transforms=tfs,
        num_classes=model_settings["num_classes"],
        threshold=0.5,
        img_folder=img_folder,
        label_folder=label_folder,
        # label_ext=".txt",
        load_true=True,
    )
    if subset_n:
        ds_inds = np.random.choice(len(base_ds), subset_n)
        base_ds = Subset(base_ds, ds_inds)
    test_dl = DataLoader(base_ds, batch_size=model_settings["bs"], num_workers=4)
    
    # 4. run inference and get results out
    trainer_args = {
        "accelerator" : "gpu",
        "devices" : [0],
        "logger": pl_loggers.TensorBoardLogger(save_dir=logdir, name="predictions")
    }
    trainer = pl.Trainer(
        **trainer_args
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None    # none needed
    # trainer.fit(subens_model, train_dataloaders=inf_dl)
    test_acc_epoch = trainer.test(subens_model, test_dl)
    return test_acc_epoch

if __name__=="__main__":

    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # get the model
    args = default_arguments(basedir)
    logdir = os.path.join(basedir, 'lightning_logs', 'subensemble')
    df = test_subensemble(args["mdl_pths"], args["dataset"], args, logdir)

    df.sort_values('entropy', inplace=True, ascending=False)
    print(df.head(20))

    