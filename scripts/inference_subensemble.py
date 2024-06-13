# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
# lightning Changelog: https://lightning.ai/docs/pytorch/stable/upgrade/from_1_7.html
# tutorial: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
import os.path
import torch
from torchvision import transforms as torch_tfs
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from alg.resnet_ae import ResnetAutoencoder
from alg.subensemble import SubEnsemble
from alg.dataloader import ALGDataset

def default_arguments():
    args = {
        "ae_model" : "RN18-256.ckpt",
        "num_classes": 1,
        "resnet" : 18,
        "entropy_threshold" : 0.3,
        "heads" : [
            "lightning_logs/subensemble/head/0.pt",
            "lightning_logs/subensemble/head/1.pt",
            "lightning_logs/subensemble/head/2.pt",
            "lightning_logs/subensemble/head/3.pt",
            "lightning_logs/subensemble/head/4.pt"
        ],
        "dataset" : "data/vegetative/test"
    }
    return args

if __name__=="__main__":

    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # get the model
    args = default_arguments()
    subens_model =   SubEnsemble(
        args["num_classes"],
        args["resnet"],
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        entropy_threshold=args["entropy_threshold"],
        heads=args["heads"]
    )

    # 1. load backbone from autoencoder
    modeldir = os.path.join(basedir, 'models', 'ae')
    modelpath = os.path.realpath(os.path.join(modeldir, args["ae_model"]))
    print("Loading autoencoder from: {}".format(modelpath))
    resn_ae = ResnetAutoencoder.load_from_checkpoint(checkpoint_path = modelpath)
    missing_keys, unexp_keys = subens_model.from_AE(resn_ae)
    print("Missing Layers: {}".format(missing_keys))
    print("Unexpected Layers: {}".format(unexp_keys))

    # 2. load the heads
    subens_model.freeze()

    # 3. load the inference dataset
    mean = (0.5,)
    std = (0.5,)
    tfs = torch_tfs.Compose([
        # torch_tfs.RandomCrop(32,32),
        # torch_tfs.PILToTensor(),
        torch_tfs.ConvertImageDtype(torch.float),
        torch_tfs.Normalize(mean, std)        
    ])
    base_ds = ALGDataset(
        root=args["dataset"],
        transforms=tfs,
        num_classes=args["num_classes"],
        threshold=0.5,
        label_ext=".txt",
    )
    inf_dl = DataLoader(base_ds, batch_size=12, num_workers=1)
    
    # 4. run inference and get results out
    logdir = os.path.join(basedir, 'lightning_logs', 'subensemble')
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
    dl_dicts = trainer.predict(subens_model, inf_dl)
    print("Test debug line")
    