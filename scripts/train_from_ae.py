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
from alg.resnet_ae import ResnetAutoencoder
from alg.dataloader import ALGDataModule
from test_model import test_model
from train_model import train_model

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

if __name__ == "__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args = parse_args(basedir)

    # root data dir
    rootdir = os.path.join(basedir, 'data')
    datadir = os.path.join(rootdir, args.datadir)
    dataset_name = args.datadir

    logdir = os.path.join(basedir, 'lightning_logs',   "binary_{}".format(args.limit), 'from_ae', dataset_name)

    # # Instantiate Model
    model = ResNetClassifier(
        num_classes=args.num_classes,
        resnet_version=args.model,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        transfer=args.transfer,
        tune_fc_only=args.tune_fc_only,
    )

    # updating with weights from Autoencoder
    modeldir = os.path.join(basedir, 'models')
    modelpath = os.path.realpath(os.path.join(modeldir, args.ae_model))
    print("Loading autoencoder from: {}".format(modelpath))
    resn_ae = ResnetAutoencoder.load_from_checkpoint(checkpoint_path = modelpath)
    missing_keys, unexp_keys = model.from_AE(resn_ae)
    print("Missing Layers: {}".format(missing_keys))
    print("Unexpected Layers: {}".format(unexp_keys))

    fn = os.path.splitext(os.path.basename(args.ae_model))[0] + str(dataset_name)

    # Getting the best model out
    best_path, logger = train_model(model, args, fn, logdir, datadir)

    test_dir = os.path.join(datadir, 'test')
    results = test_model(
        best_path,
        test_dir,
        threshold=args.threshold,
        logger=logger
    )


        