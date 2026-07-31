import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from typing import (Optional, Dict, Any, Literal, Tuple)
from torchtyping import TensorType
from transformers.cache_utils import Cache, DynamicCache


_ATTENTION_SCORING_FUNCTIONS_: Dict[str, nn.Module] = {}
def register_scoring_fn(name: str):
    def wrapper(fn: nn.Module):
        if name not in _ATTENTION_SCORING_FUNCTIONS_:
            _ATTENTION_SCORING_FUNCTIONS_[name] = fn
        else:
            raise ValueError(f"{name} scoring function is already implemented!!!")
        return fn
    return wrapper

def get_scoring_fn(name: str, **kwargs):
    if name in _ATTENTION_SCORING_FUNCTIONS_:
        return _ATTENTION_SCORING_FUNCTIONS_[name](**kwargs)
    else:
        raise ValueError(f"{name} scoring method is not implementd !!!")


class BaseAttenionScoring(nn.Module):
    def __init__(self, features: int, nheads: int):
        super(BaseAttenionScoring, self).__init__()
        self.d = features
        self.nheads = nheads
        self.d_model = int(self.d // self.nheads)

    def _check_mask(self, query, keys, mask):
        (B, _, Sq, _) = query.shape
        (_, _, Sk, _) = keys.shape
        assert mask.shape == (B, Sq, Sk) \
            or mask.shape == (1, Sq, Sk), \
            (f"""{mask.shape=} must be the same in size (B, S, S`).
             Where B, and S are batch_size and tokens_sequence_lenght
             corespondeling for query: TensorType['B', 'nh', 'S', 'C'] 
             and keys: TensorType['B', 'nh', 'S`', 'C']!!!""")
        
    def forward(self, query: TensorType["B", "nh", "S", "C"],
                keys: TensorType["B", "nh", "S", "C"],
                values: Optional[TensorType["B", "nh", "S", "C"]]=None,
                mask: Optional[TensorType["B", "S", "S"]]=None):
        raise NotImplemented("tried to use forward from base class")
    
@register_scoring_fn("scaled-dot-product")
class SDPAttention(BaseAttenionScoring):
    def forward(self, query: TensorType["B", "nh", "S", "C"],
                keys: TensorType["B", "nh", "S", "C"],
                values: Optional[TensorType["B", "nh", "S", "C"]]=None,
                mask: Optional[TensorType["B", "S", "S"]]=None):
        
        qkT = (query @ keys.transpose(-1, -2))
        qkT /= (self.d ** 0.5)
        if mask is not None:
            self._check_mask(query, keys, mask)
            qkT.masked_fill_(mask.unsqueeze(dim=1), -1e9)
        
        scores = F.softmax(qkT, dim=-1)
        values = values if values is not None else keys
        attention = (scores @ values)
        return attention
    

@register_scoring_fn("cosine-similarity")
class CosineAttention(BaseAttenionScoring):
    def forward(self, query: TensorType["B", "nh", "S", "C"],
                keys: TensorType["B", "nh", "S", "C"],
                values: Optional[TensorType["B", "nh", "S", "C"]]=None,
                mask: Optional[TensorType["B", "S", "S"]]=None):
        q_norm = th.linalg.norm(query, dim=-1, keepdim=True)
        k_norm = th.linalg.norm(keys, dim=-1, keepdim=True)
        print(query.shape, q_norm.shape, keys.shape, k_norm.shape)
        cosine = (query / q_norm) @ (keys / k_norm).transpose(-1, -2)
        if mask is not None:
            self._check_mask(query, keys, mask)
            cosine.masked_fill_(mask.unsqueeze(dim=1), -1e9)
        values = values if values is not None else keys
        print(cosine.shape, values.shape)
        attention = cosine @ values
        return attention
    
@register_scoring_fn("additive")
class AdditiveAttention(BaseAttenionScoring):
    def __init__(self, features: int, nheads: int):
        super(AdditiveAttention, self).__init__(features, nheads)
        self.qnet = nn.Linear(self.d_model, self.d_model)
        self.knet = nn.Linear(self.d_model, self.d_model)
        self.register_buffer("projection", th.zeros((self.d_model, )))
    def forward(self, query: TensorType["B", "nh", "S", "C"],
                keys: TensorType["B", "nh", "S", "C"],
                values: Optional[TensorType["B", "nh", "S", "C"]]=None,
                mask: Optional[TensorType["B", "S", "S"]]=None):
        
        (B, nh, S, C) = query.size()
        (_, _, Sk, _) = keys.size()
        assert (S == Sk), \
        (f"""for additive scoring sequence_lenghts 
         for query and keys must be the same""") 
        print(self.qnet(query).size(), self.knet(keys).size())
        QK = F.tanh(self.qnet(query) + self.knet(keys))
        scores = (self.projection\
            .view(1, 1, 1, self.d_model) @ QK.transpose(-1, -2))
        if mask is not None:
            self._check_mask(query, keys, mask)
            scores.masked_fill_(mask.unsqueeze(dim=1), -1e9)
        values = values if values is not None else keys
        attention = (scores @ values)
        return attention
        


class MultiHeadAttention(nn.Module):
    def __init__(self, features: int,
                 nheads: int,
                 scoring_fn: str="scaled-dot-product",
                 heads_reduction: Literal["sum", "mean", "w-sum"]="w-sum",
                 layer_idx: Optional[int]=None):
        
        super(MultiHeadAttention, self).__init__()
        assert (features % nheads) == 0, \
        (f"features dim: {features} must be dividable by nheads: {nheads}")
        self.layer_idx = layer_idx
        self.d = features
        self.nh = nheads
        self.d_model = int(features // nheads)
        self.hreduction = heads_reduction
        self.scoring = get_scoring_fn(scoring_fn, 
                                      features=features,
                                      nheads=nheads)
        if heads_reduction == "w-sum":
            heads_probs = th.zeros((nheads, ))
            self.register_buffer("heads_probs", heads_probs)
        (self.qnet, self.knet, self.vnet) = tuple(nn.Linear(features, features) for _ in range(3))
        self.projection = nn.Linear(self.d_model, features)
    
    def _split_heads(self, x: TensorType["B", "S", "C"]):
        (B, S, _) = x.size()
        return x.view(B, self.nh, S, self.d_model)
    
    def _merge_heads(self, x: TensorType["B", "nh", "S", "C"]):
        if self.hreduction == "sum":
            return x.sum(dim=1)
        elif self.hreduction == "mean":
            return x.mean(dim=1)
        elif self.hreduction == "w-sum":
            scores = self.heads_probs\
                    .view(1, self.nh, 1, 1)
            return (scores*x).sum(dim=1)
        else:
            raise ValueError(f"unknown heads_reduction type: {self.hreduction}")
    
    def forward(self, x: TensorType["B", "S", "C"],
                keys: Optional[TensorType["B", "S", "C"]]=None,
                values: Optional[TensorType["B", "S", "C"]]=None,
                mask: Optional[TensorType["B", "S", "S"]]=None,
                use_cache: bool=False,
                past_key_values: Optional[Cache]=None,
                cache_position: Optional[th.LongTensor]=None):
        keys = (keys if keys is not None else x)
        values = (values if values is not None else keys)
        Q = self._split_heads(self.qnet(x))
        K = self._split_heads(self.knet(keys))
        V = self._split_heads(self.vnet(values))
        present_key_values = None
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            (K, V) = K.transpose(1, 2), V.transpose(1, 2)
            (K, V) = past_key_values.update(K, V, self.layer_idx, 
                                            {"cache_position": cache_position})
            (K, V) = K.transpose(1, 2), V.transpose(1, 2)
            present_key_values = past_key_values

        attention = self.scoring(Q, K, V, mask=mask)
        attention = self._merge_heads(attention)
        attention = self.projection(attention)
        return attention, present_key_values