"""huber-ecg-base model from hf hub. 

    Huber model uses standdart hubert net model architecture
    processing waves from each channel axis in flattened manner. 
    This kind of aggregations yilds to coreelation covering on 
    level of basic conf pathces which also named 'frames' and 
    can be seen as tokens that time space was separated. 

    Though it is possible to use frames tokens format 
    BackBoneWrapper will preferably use polled representation
    as they were ment to be more general and unique representation 
    for ECG signals. 

    Model was trained on waves of 5 sec. recorded with 100 Gz frequecy,
    which means hat standart input must have dimentionality of [B, 12, 500]
    which after flattening will become [B, 6000] resulting in latent representation 
    with sequence lenght (n of tokens) of 93. 

    Thefore preprocess function is ment so chunk input raw wave sequences 
    into chunks each of needed size after which they can be processed in chuking manner 
    and tacked together. 
"""

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel
from ..backbone_registry import register_backbone, get_backbone
from torchtyping import TensorType
from typing import Optional, Callable, Iterable
from transformers import AutoModel


@register_backbone("huber-ecg-base", overwrite=True)
class HuberBaseBackbone(nn.Module):
    repo_id: str = "Edoardo-BS/hubert-ecg-base"
    embedding_dim: int = 768
    def __init__(self):
        super(HuberBaseBackbone, self).__init__()
        self._core = AutoModel.from_pretrained(self.repo_id)

    def forward(self, x: th.Tensor):
        """Huber model have padded_output field
        that models hadnles general features representations 
        about all raw signals that was passed as input.
        
        The input x: th.Tensor must have form: [B, 12, Times].
        """
        B = x.shape[0]
        x = x.flatten(start_dim=-2)
        x = self._core(x)
        return x.last_hidden_state



