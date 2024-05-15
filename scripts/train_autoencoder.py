
import os.path
import torch

import pytorch_lightning as pl
from torchvision import transforms as torch_tfs
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from alg.resnet_ae import ResnetAutoencoder
from alg.ae_utils import GenerateCallback
from alg.ae_dataloader import ALGRAWDataModule

# using torchvision only transforms: https://pytorch.org/vision/0.13/transforms.html 

if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(42)

    # model
    ae = ResnetAutoencoder(18, True, width=32, height=32)
    name = str(ae) + "_alg256_32_Ident"

    # Dataset
    # Transformations applied on each image => only make them a tensor
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mean = (0.5,)
    std = (0.5,)
    tfs = torch_tfs.Compose([
        torch_tfs.RandomCrop(32,32),
        torch_tfs.PILToTensor(),
        torch_tfs.ConvertImageDtype(torch.float),
        torch_tfs.Normalize(mean, std)        
    ])
  
    datadir = os.path.join(basedir, 'data', '256')
    train_datamod = ALGRAWDataModule(root=datadir, transforms=tfs, batch_size=256, num_workers=12)

    # Loading the training dataset. We need to split it into a training and validation part
    # pl.seed_everything(42)

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
        log_every_n_steps=5,
    )

    # trainer
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # training
    trainer.fit(ae, train_datamod)

    # testing
    # val_result = trainer.test(ae, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(ae, dataloaders=test_loader, verbose=False)
