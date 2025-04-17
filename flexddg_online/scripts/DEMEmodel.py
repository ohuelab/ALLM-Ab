import torch,math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter, scatter_sum, scatter_mean

class DEME(nn.Module):
    def __init__(self,esm_pretrain_model, mode='sum'):
        super(DEME,self).__init__()
        self.esm_pretrain_model = esm_pretrain_model
        self.mode = mode

    def forward(self,data):
        esm_x,padding_mask = to_dense_batch(data['protein'].x,data['protein'].batch)
        dense_batch,_ = to_dense_batch(data['protein'].batch,data['protein'].batch)
        esm_x[~padding_mask] = 1

        esm_logits = self.esm_pretrain_model(esm_x)['logits']
        esm_mask = esm_x==32

        esm_mask_logits = esm_logits[esm_mask]
        esm_mask_batch = dense_batch[esm_mask]

        esm_logits1 = esm_mask_logits.gather(dim=1,index=data.esm_token_idx[:,[0]])
        esm_logits2 = esm_mask_logits.gather(dim=1,index=data.esm_token_idx[:,[1]])

        if self.mode == 'sum':
            esm_logits_sum1 = scatter_sum(esm_logits1, esm_mask_batch, dim=0)
            esm_logits_sum2 = scatter_sum(esm_logits2, esm_mask_batch, dim=0)
            return esm_logits_sum2-esm_logits_sum1
        elif self.mode == 'mean':
            esm_logits_sum1 = scatter_mean(esm_logits1, esm_mask_batch, dim=0)
            esm_logits_sum2 = scatter_mean(esm_logits2, esm_mask_batch, dim=0)
            return esm_logits_sum2-esm_logits_sum1
        elif self.mode == "squared":
            batch_num = esm_mask.sum(dim=1)
            esm_logits_sum1 = scatter_sum(esm_logits1, esm_mask_batch, dim=0)/batch_num
            esm_logits_sum2 = scatter_sum(esm_logits2, esm_mask_batch, dim=0)/batch_num
            return esm_logits_sum2-esm_logits_sum1
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

ABLANG2_MASK_TOKEN = 23

class DEMEAbLang2(nn.Module):
    def __init__(self,ablang2_pretrain_model):
        super(DEMEAbLang2,self).__init__()
        self.ablang2_pretrain_model = ablang2_pretrain_model

    def forward(self,data):
        tokens,padding_mask = to_dense_batch(data['protein'].x,data['protein'].batch)
        dense_batch,_ = to_dense_batch(data['protein'].batch,data['protein'].batch)
        tokens[~padding_mask] = 1

        logits = self.ablang2_pretrain_model(tokens)
        mask = tokens==ABLANG2_MASK_TOKEN

        mask_logits = logits[mask]
        mask_batch = dense_batch[mask]

        logits1 = mask_logits.gather(dim=1,index=data.token_idx[:,[0]])
        logits2 = mask_logits.gather(dim=1,index=data.token_idx[:,[1]])

        logits_sum1 = scatter_sum(logits1, mask_batch, dim=0)
        logits_sum2 = scatter_sum(logits2, mask_batch, dim=0)
        return logits_sum2-logits_sum1
