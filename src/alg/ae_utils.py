import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

class GenerateCallback(Callback):

    def __init__(self, input_images, every_n_epochs : int = 50) -> None:
        """
            Args:
                input_imgs : which images to reconstruct periodically
                every_n_epochs : how often to reconstruct
        """
        super().__init__()
        self.input_imgs = input_images      # which images

        # only reconstruct every n epochs -> to diminish tensorboard size
        self.every_n_epochs = every_n_epochs
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch + 1 % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            
            # plotting and adding to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image(f"Reconstructions: {trainer.current_epoch}", grid, global_step = trainer.global_step)


def compare_images(img1 : torch.Tensor, img2 : torch.Tensor) -> tuple:
    """
        use to compare img1 and img2
        use with plt.imshow(grid)
    """
    loss = F.mse_loss(img2, img1, reduction="sum")
    grid = make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, range=(-1,1))
    grid = grid.permute(1,2,0)
    return grid, loss

def visualize_reconstruction(model : nn.Module, img : torch.Tensor):
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(img.to(model.device))
    reconst_imgs = reconst_imgs.cpu()
    return compare_images(img, reconst_imgs)