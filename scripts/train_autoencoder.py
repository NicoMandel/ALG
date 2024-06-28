
import os.path
from argparse import ArgumentParser
import torch

import pytorch_lightning as pl
from torchvision import transforms as torch_tfs
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from alg.resnet_ae import ResnetAutoencoder
from alg.ae_utils import GenerateCallback
from alg.ae_dataloader import ALGRAWDataModule

# using torchvision only transforms: https://pytorch.org/vision/0.13/transforms.html 

def parse_args(defdir : str):
    resdir = os.path.join(defdir, 'lightning_logs')
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "datadir", help="""Path to root data folder""", type=str
    )
    parser.add_argument(
        "output", help="""Output file location. .txt file, where the name of the best autoencoder model will be stored""", type=str
    )

    # optional arguments
    parser.add_argument("-n", "--num_epochs", help="""Number of Epochs to Run.""", type=int, default=500)
    parser.add_argument(
        "-s", "--size", help="""Size of images to use for autoencoder. Smaller than 256. defaults to 32""", type=int, default=32,
    )
    return parser.parse_args()

def train_autoencoder(size : int, datadir : str, logdir : str) -> str:
    # model
    ae = ResnetAutoencoder(18, True, width=size, height=size)
    name = str(ae) + "_alg256_{}_Ident".format(size)

    # Dataset
    # Transformations applied on each image => only make them a tensor
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mean = (0.5,)
    std = (0.5,)
    _tf = [torch_tfs.RandomCrop(size, size)] if size != 256 else []
    _tf += [
        torch_tfs.PILToTensor(),
        torch_tfs.ConvertImageDtype(torch.float),
        torch_tfs.Normalize(mean, std)
        ]
    tfs = torch_tfs.Compose(
        _tf
    )
  
    train_datamod = ALGRAWDataModule(root=datadir, transforms=tfs, batch_size=256, num_workers=20)

    # Loading the training dataset. We need to split it into a training and validation part
    # pl.seed_everything(42)

    # Logger
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name=name)
    log_imgs = torch.stack([train_datamod.default_dataset[i][0] for i in range(8)], dim=0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir,
        filename=name+"-{epoch}-{val_acc:0.2f}",
        save_weights_only=True,
        save_top_k=1
        )
    # stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=10,  #! change back to 500
        precision=32,
        logger=logger,
        # fast_dev_run=True,
        callbacks=[
            checkpoint_callback,
            # stopping_callback,
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

    best_path = checkpoint_callback.best_model_path
    return best_path

if __name__=="__main__":
    args = parse_args()
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(42)

    datadir = args.datadir
    ds_name = os.path.basename(args.datadir)
    logdir = os.path.join(basedir, 'lightning_logs', 'ae', ds_name)

    best_path = train_autoencoder(32, datadir, logdir)
    with open(args.output, "w") as f:
        f.write(best_path)
    print("Written best path to: {}".format(args.output))
    
    # testing
    # val_result = trainer.test(ae, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(ae, dataloaders=test_loader, verbose=False)
