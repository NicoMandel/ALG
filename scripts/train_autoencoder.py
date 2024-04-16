
import os.path
import torch

import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

from alg.autoencoder import Autoencoder
from alg.ae_utils import GenerateCallback
from alg.ae_dataloader import ALGRAWDataModule

if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(42)

    # model
    ae = Autoencoder()
    name = str(ae) + "_alg32"

    # Dataset
    # Transformations applied on each image => only make them a tensor
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    tfs = A.Compose([
        A.RandomCrop(32,32),
        A.Normalize((0.5,), (0.5,)),
        ToTensorV2(always_apply=True)        
    ])

    datadir = os.path.join(basedir, 'data', 'raw')
    train_datamod = ALGRAWDataModule(root=datadir, transforms=tfs, batch_size=128, num_workers=12)

    # Loading the training dataset. We need to split it into a training and validation part
    pl.seed_everything(42)

    # Logger
    logdir = os.path.join(basedir, 'lightning_logs', 'ae')
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name=name)
    log_imgs = torch.stack([train_datamod.default_dataset[i][0] for i in range(8)], dim=0)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=500,
        precision=32,
        logger=logger,
        # fast_dev_run=True,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, save_top_k=1),
            GenerateCallback(log_imgs, every_n_epochs=50),
            LearningRateMonitor("epoch")
        ],
        log_every_n_steps=2,
    )

    # trainer
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # training
    trainer.fit(ae, train_datamod)

    # testing
    # val_result = trainer.test(ae, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(ae, dataloaders=test_loader, verbose=False)
