# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
# lightning Changelog: https://lightning.ai/docs/pytorch/stable/upgrade/from_1_7.html
# tutorial: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html

import os.path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import loggers as pl_loggers

from alg.dataloader import ALGDataset
from alg.model import ResNetClassifier

def parse_args():
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "model",
        help="""Which model to test"""
    )
    parser.add_argument(
        "datadir", help="""Path to root data folder.""", type=str
    )

    # Optional arguments
    parser.add_argument(
        "-e", "--ext", help="""File extension for label files""", type=str, default=".txt"
    )
    parser.add_argument(
        "-b", "--batch_size", help="""Batch size to use during testing""", type=int, default=16
    )
    parser.add_argument(
        "-t", "--threshold", help="""Threshold when to define an image as ALG containing""", type=float, default=0.5
    )    
    parser.add_argument(
        "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=1
    )
    return parser.parse_args()

def test_model(
        model_path : str, datadir : str, threshold : float = 0.5, fext : str = ".txt", batch_size : int = 16,
        logger : pl_loggers.TensorBoardLogger = None 
    ):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformations = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
        ])  
    
    alg_test_ds = ALGDataset(
        root=datadir,
        label_ext=fext,
        threshold=threshold,
        transforms=transformations
    )
    bs = batch_size if batch_size < len(alg_test_ds) else len(alg_test_ds) // 2
    dl = DataLoader(alg_test_ds, batch_size=bs, num_workers=4, drop_last=True)
    
    # model
    model = ResNetClassifier.load_from_checkpoint(model_path)

    # logger
    if logger is None:
        logdir=os.path.dirname(os.path.dirname(model_path))
        logger = pl_loggers.TensorBoardLogger(
            # name=model.
            # TODO - get the name here from the model properties
            name="ToDo",
            save_dir=logdir
        )
    
    # trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        devices=1,
        logger=logger
    )
    trainer.logger._default_hp_metric = None    # none needed
    
    # Actual test step
    results = trainer.test(model = model, dataloaders=dl)
    
    return results



if __name__=="__main__":
    args = parse_args()
    results = test_model(
        args.model,
        args.datadir,
        args.threshold,
        args.ext,
        args.batch_size
    )
