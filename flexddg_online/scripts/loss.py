import torch,math
from torch import nn
from torch.nn import functional as F
import numpy as np

class myloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.margin = 0.1

    def forward(self,y):
        return self.relu(-y+self.margin).mean()

def listMLE(y_pred, y_true, eps=1e-8):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    # random_indices = torch.randperm(y_pred.shape[-1])
    # y_pred_shuffled = y_pred[:, random_indices]
    # y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true.sort(descending=True, dim=0)

    preds_sorted_by_true = torch.gather(y_pred, dim=0, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=0, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[0]), dim=0).flip(dims=[0])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    return observation_loss.mean()