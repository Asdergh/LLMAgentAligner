import torch as th
import torch.nn as nn
import transformers as tfs
import transformers.modeling_utils as tfm
from typing import (Optional, Callable, Dict, List)
from warnings import warn
from functools import wraps

from .attention import (_ATTENTION_SCORING_FUNCTIONS_, MultiHeadAttention)




def get_activation(act: str, **kwargs) -> Callable[..., nn.Module]:
    activations = {
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'leaky': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'celu': nn.CELU,
        'gelu': nn.GELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'softmax': nn.Softmax,
        'softplus': nn.Softplus,
        'softsign': nn.Softsign,
        'silu': nn.SiLU,
        'swish': nn.SiLU,
        'mish': nn.Mish,
        'hardswish': nn.Hardswish,
        'hardsigmoid': nn.Hardsigmoid,
        'hardtanh': nn.Hardtanh,
        'logsoftmax': nn.LogSoftmax,
        'identity': nn.Identity,
    }
    act = act.lower().strip()
    if act not in activations:
        warn(f"Unknown activation: {act}. Available: {list(activations.keys())}")
        return nn.Identity()
    return activations[act](**kwargs)


class Lambda(nn.Module):
    def __init__(self, fn: callable):
        super(Lambda, self).__init__()
        self.fn = fn
    
    def forward(self, x: th.Tensor, **kwargs):
        return self.fn(x, **kwargs)
    
class Mlp(nn.Module):
    def __init__(self, features: int,
                 out_features: Optional[int]=None,
                 dropout: float=0.0,
                 activation: str="relu"):
        super(Mlp, self).__init__()
        out_features = out_features if out_features is not None else features
        self.net = nn.Sequential(*[nn.Linear(features, 2*out_features),
                                   get_activation(activation),
                                   nn.Dropout(dropout),
                                   nn.Linear(2*out_features, out_features),
                                   get_activation(activation)])
    def forward(self, x: th.Tensor):
        return self.net(x)

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super(Residual, self).__init__()
        self.flow = module
    
    def forward(self, x: th.Tensor):
        x_out = self.flow(x)
        if x_out.shape == x.shape:
            return x + x_out
        else:
            warn("flow changed input tensors's size")
            return x_out
        
class FiltrationBlock(nn.Module):
    def __init__(self, features: int):
        super(FiltrationBlock, self).__init__()
        self.net = nn.Sequential(nn.Linear(features, 2), nn.Sigmoid())
    
    def forward(self, x: th.Tensor):
        scale, shift = self.net(x).chunk(2, dim=-1)
        scale = scale.view(*x.shape[:-1], -1)
        shift = shift.view(*x.shape[:-1], -1)
        return (scale * x) + shift

class Block(nn.Module):
    def __init__(self, features: int,
                 depth: int=3,
                 dropout: float=0.0,
                 filtration: bool=True,
                 n_attn_heads: int=4,
                 attention_reduction: str="w-sum",
                 attention_scoring_fn: str="scaled-dot-product",
                 residual_connections: bool=False):
        
        super(Block, self).__init__()
        self.blocks: List[nn.Module] = []
        for _ in range(depth):
            block = nn.Sequential(Mlp(features, None, dropout, "identity"),
                                MultiHeadAttention(features,
                                                   nheads=n_attn_heads,
                                                   scoring_fn=attention_scoring_fn,
                                                   heads_reduction=attention_reduction),
                                Mlp(features, None, dropout, "identity"))
            if filtration:
                block.append(FiltrationBlock(features))
            if residual_connections:
                block = Residual(block)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x: th.Tensor):
        for block in self.blocks:
            x = block(x)
        return x

if __name__ == "__main__":

    FEATURES = 312
    ATTN_HEADS = 4
    B, S = (10, 256)

    q = th.normal(0, 1, (B, S, FEATURES))
    k = th.normal(0, 1, (B, S, FEATURES))
    v = th.normal(0, 1, (B, S, FEATURES))
    
    block = Block(FEATURES, n_attn_heads=ATTN_HEADS,
                  dropout=0.45)
    Nparams = sum([p.numel() for p in block.parameters()])
    x = block(q)
    print(f"Total params count: {Nparams}")
    print(x.shape, x.mean())

        
        



