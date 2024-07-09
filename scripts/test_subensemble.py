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

from alg.model import ResNetClassifier
from alg.resnet_ae import ResnetAutoencoder
from alg.subensemble import SubEnsemble
from alg.subensemble_dataset import SubensembleDataset

def default_arguments(basedir : str):
    subens_file = os.path.join(basedir, "se_files_2.txt")
    with open(subens_file, 'r') as f:
        lines= [line.rstrip() for line in f]
    
    args = {
        "mdl_pths" : lines,
        "num_classes": 1,
        # "resnet" : 18,
        "bs" : 32,
        "dataset" : "data/combined",     # /test 
        "resnet_version" : 18      
    }
    return args

def test_subensemble(mode: str, mdl_pths : list, dataset_path : str, model_settings : dict, logdir : str,
                          img_folder : str = "images", label_folder : str = "labels", load_true : bool=False, from_ae : bool = True,
                          subset_n : int = None) -> float | pd.DataFrame:
    """
        mode should be either "test" or "inference"
    """
    subens_model =   SubEnsemble(
        model_settings["num_classes"],
        model_settings["resnet_version"],
        transfer=True,
        heads=mdl_pths[1:],
        load_true = load_true
    )
    print("Loading backbone from: {}".format(mdl_pths[0]))
    if from_ae:
        resn_ae = ResnetAutoencoder.load_from_checkpoint(checkpoint_path = mdl_pths[0])
        missing_keys, unexp_keys = subens_model.from_AE(resn_ae)
        if missing_keys or unexp_keys: print("Missing Layers: {},\n\nUnexpected Layers: {}".format(missing_keys, unexp_keys))
    else:
        resn = ResNetClassifier.load_from_checkpoint(checkpoint_path = mdl_pths[0])
        missing_keys, unexp_keys = subens_model.from_classif(resn)
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
        load_true=load_true,
    )
    if subset_n:
        ds_inds = np.random.choice(len(base_ds), subset_n)
        base_ds = Subset(base_ds, ds_inds)
    dl = DataLoader(base_ds, batch_size=model_settings["bs"], num_workers=4)
    
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
    if mode == "test":
        test_acc_epoch = trainer.test(subens_model, dl)
        return test_acc_epoch
    
    elif mode == "inference":
        results = trainer.predict(subens_model, dl)
        dld = {}
        [dld.update(a) for a in results]
        columns = ["vote", "class_indices", "entropy"]
        if load_true: columns.append("label")
        df = pd.DataFrame.from_dict(dld, orient='index', columns=columns)
        return df
    else: raise TypeError("Mode should either be inference or test, not: {}".format(mode))


if __name__=="__main__":

    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # get the model
    args = default_arguments(basedir)
    logdir = os.path.join(basedir, 'lightning_logs', 'set')
    df = test_subensemble("inference", args["mdl_pths"], args["dataset"], args, logdir, load_true=True, subset_n=2000)

    df.sort_values('entropy', inplace=True, ascending=False)
    print(df.head(20))

    