import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class ResnetEncoder(nn.Module):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, resnet_version, transfer=True):
        super().__init__()
        net = self.resnets[resnet_version](pretrained=transfer)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.reshape(x.shape[0], -1)
    
    def linear_size(self):
        bbsq = list(self.net.children())[-3]
        bb = list(bbsq.children())[-1]
        ft = list(bb.children())[-1].num_features
        return ft

class DecoderBlock(nn.Module):
    """
        A Decoder block that effectively does a convolution with a 3x3 kernel and then doubles the resolution.
        Middel is passing a GELU layer.
        Transposed layers:
            * [explanation](https://d2l.ai/chapter_computer-vision/transposed-conv.html)
            * [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
    """

    def __init__(self, in_channels : int, out_dim : int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, out_dim, kernel_size=3, output_padding=1, padding=1, stride=2),      # 16 x 16 => 32 x 32
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class Dec(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        # B x C x H x W
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

class Decoder256(Dec):

    def __init__(self, num_input_channels : int = 3, c_hid : int = 32, latent_dim : int =512) -> None:
        """
            args:
                num_input_channels : Number of Channels of the image to reconstruct. For CIFAR, parameter is 3.
                c_hid : base_channel size - number of channels to use in the last convolutional layer. Duplicates may be used in earlier layers. default 32
                latent_dim : dimensionality of latent representation z - default 512, can be multiple of 2 from 64 on.
                act_fn : activation function used throughout encoder network. Default is GELU
        """
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 4 * 4 * c_hid), nn.GELU())
        # 16 x 16 is from the reshape into a -1, 4, 4 ! -> so this effectively turns the size into 16 times that
        # input here is a B x 64 x 4 x 4
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 64 x 4 x 4 => 64 x 8 x 8
            nn.GELU(),
            DecoderBlock(2 * c_hid, c_hid), # 64 x 8 x 8 [4096]  -> 32 x 16 x 16 [8192]
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32 x 16 x 16 => 32 x 32 x 32
            nn.GELU(),
            DecoderBlock(c_hid, c_hid // 2),  # 32 x 32 x 32 -> 16 x 64 x 64 [65536]
            nn.GELU(),
            DecoderBlock(c_hid // 2, c_hid // 4),  # 16 x 64 x 64 [65536] -> 8 x 128 x 128
            nn.GELU(),
            DecoderBlock(c_hid // 4, num_input_channels),  # 8 x 128 x 128 [131072] -> 3 x 256 x 256 [196608] 
            nn.Tanh(),
        )

class Decoder32(Dec):

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

class ResnetAutoencoder(pl.LightningModule):

    def __init__(self,
                 resnet_version : int,
                 transfer : bool = True,
                 c_hid : int = 32,
                 num_input_channels : int = 3,
                 width : int = 32,
                 height : int = 32,
                 lr : int = 1e-3,
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
        self._lr = lr
        self._res_ver = resnet_version
        self.save_hyperparameters()

        self.encoder = ResnetEncoder(resnet_version, transfer)
        lin_feat = self.encoder.linear_size()
        self.latent_dim = lin_feat
        if width == 256:
            self.decoder = Decoder256(num_input_channels, c_hid, lin_feat)
        elif width == 32:
            self.decoder = Decoder32(num_input_channels, c_hid, lin_feat)
        else:
            raise ValueError("Unknown width {} for Autoencoder training. Possible dimensions are: {}".format(width, (32, 256)))
        self.example_input_array = torch.zeros(1, num_input_channels, width, height)
        
        self.loss = nn.MSELoss(reduction="none")

    def __str__(self) -> str:
        return f"ResNet{self._res_ver}AE:dec:{len(self.decoder.net) + 1}-lat:{self.encoder.linear_size()}"

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
            Loss defined differently, see nn module fct. x and x_hat are inverted
        """
        x, y = batch        
        x_hat = self.forward(x)
        loss = self.loss(x_hat, y)
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self._lr)

        # scheduler is optional but helpful
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=60, min_lr=5e-5)
        return {"optimizer" : optimizer, "lr_scheduler" : scheduler, "monitor" : "val_loss"}
        # return optimizer

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
    