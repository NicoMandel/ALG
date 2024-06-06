import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
import numpy as np
import pytorch_lightning as pl
from alg.resnet_ae import ResnetAutoencoder

EPSILON = 1e-7
EPSILON2 = 1e-10

def _negative_log_likelihood_pt(y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    y_pred = torch.clamp(y_pred, EPSILON, 1.0-EPSILON)
    nll = -torch.mean(torch.sum( y_true * torch.log(y_pred) + 1.0 - y_true * torch.log(1.0 - y_pred), axis=-1), axis=-1)
    return nll
    

def _negative_log_likelihood_np(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    nll = -np.mean(np.sum( y_true * np.log(y_pred) + 1.0 - y_true * np.log(1.0 - y_pred), axis=-1), axis=-1)
    return nll

def negative_log_likelihood(y_true : np.ndarray | torch.Tensor, y_pred : np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    assert isinstance(y_pred, type(y_true)), "Not equal types. Y true: {}, while Y pred: {}".format(type(y_true), type(y_pred))
    if isinstance(y_pred, np.ndarray):
        return _negative_log_likelihood_np(y_true, y_pred)
    else:
        return _negative_log_likelihood_pt(y_true, y_pred)
    
def compute_entropy(data, axis=-1):
    if isinstance(data, np.ndarray):
        cls = np
    else:
        cls = torch
    return cls.sum(-data * cls.log(data + EPSILON2), axis=axis)

class SubEnsemble(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        resnet_version,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        entropy_threshold : float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version](pretrained=transfer)
        self.old_fc = self.resnet_model.fc.copy()
        self.resnet_model.fc = nn.Identity()

        # for visulising the model, an example input array is needed
        self.example_input_array = torch.zeros(2,3,256,256)
        self.save_hyperparameters()

    def freeze_layers(self):
        """
            Function to freeze all layers -> call after loading from ae
        """
        for child in list(self.resnet_model.children()):
            for param in child.parameters():
                param.requires_grad = False

    def freeze_heads(self):
        for fcl in self.fc_layers:
            for param in fcl.parameters():
                param.requires_grad = False

    def forward(self, x):
        embed =  self.resnet_model(x)
        y_pred = torch.stack([F.softmax(fcl(embed)) for fcl in self.fc_layers], dim=1)
        return y_pred

    def configure_optimizers(self):
        """
            Unnecessary for inference
        """
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc
    
    def predict_step(self, batch: F.Any, batch_idx: int, dataloader_idx: int = 0) -> F.Any:
        x, fname = batch
        y_pred = self.forward(x)
        entr = compute_entropy(y_pred)
        return list(zip(fname, entr))
    
    def from_AE(self, AE_model : ResnetAutoencoder):
        """
            Function to update the weights to that of a corresponding Autoencoder 
        """
        new_dict = AE_model.encoder.net.state_dict()
        missing_keys, unexpected_keys = self.resnet_model.load_state_dict(new_dict, strict = False)
        self.freeze_layers()
        return missing_keys, unexpected_keys
    
    def from_resnets(self, *resnet_models):
        fc_layers = []
        for mdl in resnet_models:
            fc_l = mdl.fc
            ll_n = nn.Linear(512, self.num_classes)
            m_k, un_k = ll_n.load_state_dict(fc_l.state_dict(), strict=False)
            print(f"Missing keys: {m_k}, unknown keys: {un_k}")
            ll_n.eval()
            fc_layers.append(ll_n)
        
        self.fc_layers = fc_layers
