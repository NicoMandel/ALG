# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
# lightning Changelog: https://lightning.ai/docs/pytorch/stable/upgrade/from_1_7.html
# tutorial: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
import warnings
import os.path
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import loggers as pl_loggers

from alg.model import ResNetClassifier
from alg.utils import model_settings_from_args
from alg.dataloader import ALGDataModule
from test_model import test_model

# from copy import deepcopy

def parse_args(defdir : str):
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
        "-tr",
        "--transfer",
        help="""Determine whether to use pretrained model or train from scratch. Defaults to from scratch.""",
        action="store_true",
    )
    parser.add_argument(
        "-to",
        "--tune_fc_only",
        help="Tune only the final, fully connected layers. Defaults to false",
        action="store_true",
    )
    parser.add_argument(
        "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=1
    )
    parser.add_argument(
        "--limit", help="""Limit Training and validation Batches - how much data to use as a subset.""", type=float, default=1.0
    )
    return parser.parse_args()

def get_augmentations(p : float = 0.5, mean : list = [0.485, 0.456, 0.406], std : list = [0.229, 0.224, 0.225]) -> A.Compose:
    augmentations = A.Compose([
        A.OneOf([
            A.VerticalFlip(p=p),
            A.Rotate(limit=179, p=p),
            A.HorizontalFlip(p=p)
        ], p=1),
        # Color Transforms
        A.OneOf([ 
            # A.CLAHE(),
            A.RandomBrightnessContrast(p=p),
            A.RandomGamma(p=p),
            A.HueSaturationValue(p=p),
            A.GaussNoise(p=p)
        ], p=1),
        # Elastic Transforms
        A.OneOf([
            A.ElasticTransform(p=p),
            A.GridDistortion(p=p),
            A.OpticalDistortion(p=p),
        ], p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()        # 
    ])
    return augmentations

def train_model(model : pl.LightningModule, model_settings :  dict, fn : str, logdir, datadir, seed : int = 42):
    # Set up Datamodule - with augmentations
    augmentations = get_augmentations()
    datamodule = ALGDataModule(
        root = datadir,
        img_folder="images",
        label_folder="labels",
        transforms=augmentations,
        batch_size=model_settings["bs"], 
        num_workers=4,
        threshold=model_settings["threshold"],
        limit=model_settings["limit"],
        val_percentage=0.2,
        img_ext=".tif",
        label_ext=".tif",
        seed=seed
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=logdir,
        filename=fn+"-{epoch}-{val_acc:0.2f}",
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=20)

    # Instantiate lightning trainer and train model
    trainer_args = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": [0],
        "strategy": "dp" if torch.cuda.is_available() else None,
        "max_epochs": model_settings["num_epochs"],
        "callbacks": [checkpoint_callback, stopping_callback],
        "precision": 32,
        "logger": pl_loggers.TensorBoardLogger(save_dir=logdir, name=fn),
        # "fast_dev_run" : True
    }
    trainer = pl.Trainer(**trainer_args)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None    # none needed

    trainer.fit(model, datamodule=datamodule)

    # Getting the best model out
    best_path = checkpoint_callback.best_model_path
    print(f"Best model at: {best_path}")
    return best_path, trainer.logger

if __name__ == "__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ms = parse_args(basedir)
    model_settings = model_settings_from_args(ms)

    # root data dir
    rootdir = os.path.join(basedir, 'data')
    datadir = os.path.join(rootdir, model_settings["datadir"])
    dataset_name = model_settings["datadir"]

    logdir = os.path.join(basedir, 'lightning_logs', "binary_{}".format(model_settings["limit"]), dataset_name)

    # # Instantiate Model
    model = ResNetClassifier(
        num_classes=model_settings["num_classes"],
        resnet_version=model_settings["model"],
        optimizer=model_settings["optim"],
        lr=model_settings["lr"],
        batch_size=model_settings["bs"],
        transfer=model_settings["transfer"],
        tune_fc_only=model_settings["tune_fc_only"],
    )

    mdl_config = ""
    if model_settings["transfer"]:
        mdl_config += "-transfer"
    if model_settings["tune_fc_only"]:
        mdl_config += "-finetune"
    fn = "resnet{}-{}".format(model_settings["model"], mdl_config) + "-" + str(dataset_name)
    best_path, logger = train_model(model, model_settings, fn, logdir, datadir)

    test_dir = os.path.join(datadir, 'test')
    results = test_model(
        best_path,
        test_dir,
        threshold=model_settings["threshold"],
        logger=logger
    )


        