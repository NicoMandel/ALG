import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, num_input_channels : int = 3, c_hid : int = 32, latent_dim : int = 512, act_fn : nn.Module = nn.GELU) -> None:
        """
            num_input_channels : Number of input channels of the image. For CIFAR, is 3
            c_hid : base_channel_size : Number of channels in first conv layer. deeper layers duplicate this - default 32
            latent_dim : dimensionality of latent representation z - default 512, can be multiple of 2 from 64 on.
            act_fn : activation function used throughout encoder network. Default is GELU
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size = 3, padding =1, stride = 2),      # 32 x 32 => 16 x 16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),                                  # 16 x 16 => 16 x 16
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),                    # 16 x 16 => 8 x 8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),                # 8 x 8 => 4 x 4
            act_fn(),
            nn.Flatten(),
            nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class Decoder(nn.Module):

    def __init__(self, num_input_channels : int = 3, c_hid : int = 32, latent_dim : int  =512, act_fn : nn.Module = nn.GELU) -> None:
        """
            args:
                num_input_channels : Number of Channels of the image to reconstruct. For CIFAR, parameter is 3.
                c_hid : base_channel size - number of channels to use in the last convolutional layer. Duplicates may be used in earlier layers. default 32
               latent_dim : dimensionality of latent representation z - default 512, can be multiple of 2 from 64 on.
                act_fn : activation function used throughout encoder network. Default is GELU
        """
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4 x 4 => 8 x 8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),      #  8 x 8 => 16 x 16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),      # 16 x 16 => 32 x 32
            nn.Tanh(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

class Autoencoder(pl.LightningModule):

    def __init__(self,
                 c_hid : int = 32,
                 latent_dim  : int =512,
                 num_input_channels : int = 3,
                 width : int = 32,
                 height : int = 32,
                 encoder_class : nn.Module = Encoder,
                 decoder_class : nn.Module = Decoder,
                *args: pl.Any, **kwargs: pl.Any) -> None:
        """
            AutoEncoder Architecture.
            num_input_channels : Number of input channels of the image. For CIFAR, is 3
            c_hid : base_channel_size : Number of channels in first conv layer. deeper layers duplicate this - default 32
            latent_dim : dimensionality of latent representation z - default 512, can be multiple of 2 from 64 on.
            Encoder and Decoder are the architectures
            width and height are dimensions of input. to set the graph
        """
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.save_hyperparameters()

        self.encoder = encoder_class(num_input_channels, c_hid, latent_dim)
        self.decoder = decoder_class(num_input_channels, c_hid, latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        
        self.loss = nn.MSELoss(reduction="none")

    def __str__(self) -> str:
        return f"enc:{len(self.encoder.net)}-dec:{len(self.decoder.net) + 1}-lat:{self.latent_dim}"

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
            Loss defined differently, see nn module fct. x and x_hat are inverted
        """
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss(x_hat, x)
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # scheduler is optional but helpful
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer" : optimizer, "lr_scheduler" : scheduler, "monitor" : "val_loss"}

    # steps
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
    