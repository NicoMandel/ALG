import numpy as np
import torch
from torch import nn

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
    
def compute_entropy(data, axis=-1, binary : bool = False):
    if binary:
        return _compute_binary_entropy(data)
    if isinstance(data, np.ndarray):
        cls = np
    else:
        cls = torch
    return cls.sum(-data * cls.log(data + EPSILON2), axis=axis)

def _compute_binary_entropy(data : torch.Tensor | np.ndarray):
    if isinstance(data, np.ndarray):
        cls = np
    else:
        cls = torch
    return -data * cls.log(data + EPSILON2)

class ParallelModule(nn.Sequential):
    def __init__(self, *args):
        super(ParallelModule, self).__init__(*args)
    
    def forward(self, input):
        output = []
        for module in self:
            output.append(module(input))
        return torch.cat(output, dim=1)
