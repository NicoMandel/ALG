
import os.path
import torch

import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from alg.autoencoder import Autoencoder
from alg.ae_utils import GenerateCallback

if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(42)

    # model
    ae = Autoencoder()
    name = str(ae)

    # Dataset
    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root="~/data", train=True, transform=transform)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root="~/data", train=False, transform=transform)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    # Logger
    logdir = os.path.join(basedir, 'lightning_logs', 'ae')
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name=name)
    log_imgs = torch.stack([train_dataset[i][0] for i in range(8)], dim=0)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=500,
        precision=32,
        fast_dev_run=True,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, save_top_k=1),
            GenerateCallback(log_imgs, every_n_epochs=50),
            LearningRateMonitor("epoch")
        ]
    )

    # trainer
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # training
    trainer.fit(ae, train_loader, val_loader)

    # testing
    val_result = trainer.test(ae, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(ae, dataloaders=test_loader, verbose=False)
