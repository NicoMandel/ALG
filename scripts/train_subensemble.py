# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
# lightning Changelog: https://lightning.ai/docs/pytorch/stable/upgrade/from_1_7.html
# tutorial: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
import os.path
from argparse import ArgumentParser
import torch
import numpy as np
from torchvision import transforms as torch_tfs
from torch.utils.data import random_split, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from copy import deepcopy

from alg.model import ResNetClassifier 
from alg.resnet_ae import ResnetAutoencoder
from alg.dataloader import ALGDataset

def parse_args(defdir):
    resdir = os.path.join(defdir, 'lightning_logs')
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "model",
        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
        type=int,
        default=18
    )
    parser.add_argument(
        "num_classes", help="""Number of classes to be learned.""", type=int, default=1,
    )
    parser.add_argument("num_epochs", help="""Number of Epochs to Run.""", type=int, default=300)
    parser.add_argument(
        "datadir", help="""Path to root data folder, as child folder of ../data .""", type=str
    )

    parser.add_argument(
        "ae_model", help="""Path to ae model which to use for transfer, as sub-path from the ./models directory""", type=str
    )

    # optional arguments
    parser.add_argument(
        "-o",
        "--optimizer",
        help="""PyTorch optimizer to use. Defaults to adam.""",
        default="adam",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Adjust learning rate of optimizer. Default 1/1000",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="""Manually determine batch size. Defaults to 16.""",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--threshold", help="""Threshold to use when to classify an image as ALG""", default=0.5, type=float
    )
    parser.add_argument(
        "--limit", help="""Limit Training and validation Batches - how much data to use as a subset.""", type=float, default=None
    )
    return parser.parse_args()

if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args = parse_args(basedir)

    # root data dir
    rootdir = os.path.join(basedir, 'data')
    datadir = os.path.join(rootdir, args.datadir)
    dataset_name = args.datadir

    logdir = os.path.join(basedir, 'lightning_logs',   "subensemble") # , dataset_name)
    # 1. initialise the backbone model
     # # Instantiate Model
    model = ResNetClassifier(
        num_classes=args.num_classes,
        resnet_version=args.model,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        tune_fc_only=True
    )
    base_fc = deepcopy(model.resnet_model.fc)

    # updating with weights from Autoencoder
    modeldir = os.path.join(basedir, 'models',  'ae')
    modelpath = os.path.realpath(os.path.join(modeldir, args.ae_model))
    print("Loading autoencoder from: {}".format(modelpath))
    resn_ae = ResnetAutoencoder.load_from_checkpoint(checkpoint_path = modelpath)
    missing_keys, unexp_keys = model.from_AE(resn_ae)
    print("Missing Layers: {}".format(missing_keys))
    print("Unexpected Layers: {}".format(unexp_keys))

    # Trainer arguments
    fn="testsomething"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=logdir,
        filename=fn+"-{epoch}-{val_acc:0.2f}",
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        save_last=True,
    )
    stopping_callback = pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=50)

    trainer_args = {
        "accelerator": "gpu",
        "devices": [0],
        "strategy": None,
        "max_epochs": args.num_epochs,
        "callbacks": [checkpoint_callback, stopping_callback],
        "precision": 32,
        "logger": pl_loggers.TensorBoardLogger(save_dir=logdir, name=fn),
        "fast_dev_run" : True
    }
    trainer = pl.Trainer(**trainer_args)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None    # none needed

    model.freeze_backbone()
    # 2. initialise a dataset for training
    mean = (0.5,)
    std = (0.5,)
    tfs = torch_tfs.Compose([
        # torch_tfs.RandomCrop(32,32),
        # torch_tfs.PILToTensor(),
        torch_tfs.ConvertImageDtype(torch.float),
        torch_tfs.Normalize(mean, std)        
    ])
    base_ds = ALGDataset(
        root=args.datadir,
        transforms=tfs,
        num_classes=2,
        threshold=0.5
    )

    # Subsampling the dataset to only hold some percentage of training data - low volume!
    np.random.seed(0)       # to always get the same subset!
    if args.limit:
        ds_count = int(np.floor(args.limit * len(base_ds)))
        ds_inds = np.random.choice(len(base_ds), ds_count)
        base_ds = Subset(base_ds, ds_inds)

    num_ens = 5
    for i in range(num_ens):
        # reset head:
        model.resnet_model.fc = base_fc
        model.freeze_backbone()

        # split training and validation dataset
        train_len = int(np.floor(0.7 * len(base_ds)))
        val_len = len(base_ds) - train_len
        generator = torch.Generator().manual_seed(i)
        train_ds, val_ds = random_split(base_ds, [train_len, val_len], generator=generator)

        # dataloaders
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=1)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=1, drop_last=True)

        # train the model
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        model.save_fc(str(i), '.')
        best_path = checkpoint_callback.best_model_path
        print(f"Best model at: {best_path}")
        best_model = ResNetClassifier.load_from_checkpoint(best_path)
        
        # store the heads 
        best_model.save_fc(i)
