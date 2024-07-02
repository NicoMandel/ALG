import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
import pytorch_lightning as pl

from alg.model import ResNetClassifier
from alg.resnet_ae import ResnetAutoencoder
from alg.subensemble_utils import ParallelModule, compute_entropy

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
        transfer=True,
        # entropy_threshold : float = 0.3,
        heads : list = None,
        load_true : bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr

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
        # self.old_fc = self.resnet_model.fc.copy()
        self.resnet_model.fc = nn.Identity()

        self.from_resnets(*heads)
        self.load_true = load_true
        # for visulising the model, an example input array is needed
        self.example_input_array = torch.zeros(2,3,256,256)
        self.save_hyperparameters()

    def linear_size(self):
        bbsq = list(self.resnet_model.children())[-3]
        bb = list(bbsq.children())[-1]
        ft = list(bb.children())[-1].num_features
        return ft

    def freeze(self) -> None:
        super().freeze()
        self.freeze_layers()
        self.freeze_heads()

    def freeze_layers(self):
        """
            Function to freeze all backbone layers -> call after loading from ae
        """
        for child in list(self.resnet_model.children()):
            for param in child.parameters():
                param.requires_grad = False
    
    def unfreeze_layers(self):
        """
            Function to freeze all backbone layers -> call after loading from ae
        """
        for child in list(self.resnet_model.children()):
            for param in child.parameters():
                param.requires_grad = True

    def freeze_heads(self):
        for fcl in self.fc_layers:
            for param in fcl.parameters():
                param.requires_grad = False

    def forward(self, x):
        # self._check_device_fcl(x)
        embed =  self.resnet_model(x)
        if self.num_classes == 1:
            y_pred = torch.stack([(fcl(embed).sigmoid()) for fcl in self.fc_layers], dim=1).squeeze()
        else:
            y_pred = torch.stack([F.softmax(fcl(embed), dim=1) for fcl in self.fc_layers], dim=1)
        return y_pred

    def configure_optimizers(self):
        """
            Unnecessary for inference
        """
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, info = batch
        if self.load_true:
            fname, label = info
        # y, fname =  info
        y_pred = self.forward(x)    
        cls_inds = (y_pred>0.5).int()    # https://stackoverflow.com/questions/58002836/pytorch-1-if-x-0-5-else-0-for-x-in-outputs-with-tensors
        vote = torch.mode((y_pred>0.5).int() , dim=1).values         # https://stackoverflow.com/questions/67510845/given-multiple-prediction-vectors-how-to-efficiently-obtain-the-label-with-most
        entr = compute_entropy(y_pred, axis=1)
        if self.load_true:
            ret_d = dict(zip(fname, zip(vote.detach().cpu().numpy(), cls_inds.detach().cpu().numpy(), entr.detach().cpu().numpy(), label.detach().cpu().numpy())))
            acc = self.acc(vote.detach().cpu().float(), label.detach().cpu())
            # self.log("inference_acc", acc, on_step=True, prog_bar=True, logger=True)
        else:
            ret_d = dict(zip(fname, zip(vote.detach().cpu().numpy(), cls_inds.detach().cpu().numpy(), entr.detach().cpu().numpy())))
        # return list(zip(fname, y, vote, cls_inds, entr))       
        return ret_d
    
    def test_step(self, batch, batch_idx : int, dataloader_idx : int = 0):
        x, info = batch
        # if self.load_true: # always true
        _, y = info
        y_pred = self.forward(x)
        vote = torch.mode((y_pred>0.5).int() , dim=1).values
        acc = self.acc(vote.detach().cpu().float(), y.detach().cpu())
        # perform logging
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)

    def from_AE(self, AE_model : ResnetAutoencoder):
        """
            Function to update the weights to that of a corresponding Autoencoder 
        """
        new_dict = AE_model.encoder.net.state_dict()
        missing_keys, unexpected_keys = self.resnet_model.load_state_dict(new_dict, strict = False)
        self.freeze_layers()
        return missing_keys, unexpected_keys
    
    def from_classif(self, resnet_model : ResNetClassifier):
        new_dict = resnet_model.resnet_model.state_dict()
        missing_keys, unexpected_keys = self.resnet_model.load_state_dict(new_dict, strict = False)
        self.freeze_layers()
        return missing_keys, unexpected_keys

    
    def from_resnets(self, *resnet_models):
        fc_layers = []
        lin_size = self.linear_size()
        for mdl in resnet_models:
            print("Loading head from {}".format(mdl))
            ll_n = nn.Linear(lin_size, self.num_classes)
            # n_fc nn.Linear().from_checkpoint()
            m_k, un_k = ll_n.load_state_dict(torch.load(mdl), strict=False)
            if m_k or un_k:  print(f"Missing keys: {m_k}, unknown keys: {un_k}")
            fc_layers.append(ll_n)
        self.fc_layers = ParallelModule(*fc_layers)
