import torch as th
import torch.nn as nn
import transformers as tfs
import transformers.modeling_utils as tfm
from typing import (Optional, Callable, Dict, List, Tuple)
from warnings import warn
from functools import wraps
from dataclasses import dataclass, asdict
from torchtyping import TensorType

from .attention import (_ATTENTION_SCORING_FUNCTIONS_, MultiHeadAttention, Cache)



@dataclass
class BlockOutput:
    features: Optional[th.Tensor]=None
    past_key_values: Optional[Cache]=None

@dataclass
class BlockStackOutput:
    last_features: Optional[th.Tensor]=None
    hidden_features: Optional[th.Tensor | Tuple[th.Tensor]]=None
    past_key_values: Optional[Cache]=None

def get_activation(act: str, **kwargs) -> Callable[..., nn.Module]:
    activations = {
        'relu':         nn.ReLU,
        'relu6':        nn.ReLU6,
        'leaky':        nn.LeakyReLU,
        'prelu':        nn.PReLU,
        'elu':          nn.ELU,
        'selu':         nn.SELU,
        'celu':         nn.CELU,
        'gelu':         nn.GELU,
        'sigmoid':      nn.Sigmoid,
        'tanh':         nn.Tanh,
        'softmax':      nn.Softmax,
        'softplus':     nn.Softplus,
        'softsign':     nn.Softsign,
        'silu':         nn.SiLU,
        'swish':        nn.SiLU,
        'mish':         nn.Mish,
        'hardswish':    nn.Hardswish,
        'hardsigmoid':  nn.Hardsigmoid,
        'hardtanh':     nn.Hardtanh,
        'logsoftmax':   nn.LogSoftmax,
        'identity':     nn.Identity,
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
    def __init__(self, module: nn.Module, 
                alpha_gating: bool=False):
        super(Residual, self).__init__()
        self.flow = module
        if alpha_gating:
            self.register_buffer("alpha", th.zeros((1, )))

    def _checked_skip(self, xout, xin):
        if xout.shape != xin.shape:
            warn("flow changed input tensors's size")
            return xout
        else:
            alpha1 = (self.alpha if hasattr(self, "alpha") else 1)
            alpha2 = (1 - self.alpha if hasattr(self, "alpha") else 1)
            return (alpha1 * xout) + (alpha2 * xin) 
        
    def forward(self, x: th.Tensor):
        xflow = self.flow(x)
        if isinstance(xflow, BlockOutput):
            x = self._checked_skip(xflow.features, x)
            return BlockOutput(x, xflow.past_key_values)
        else:
            return self._checked_skip(xflow, x)
        
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
                dropout: float=0.0,
                activation: str="gelu",
                attention_heads: int=3,
                attention_reduction: str="w-sum",
                attention_scoring_fn: str="scaled-dot-product",
                filtration: bool=True,
                layer_idx: Optional[int]=None):
        
        super(Block, self).__init__()
        self.fcin = Mlp(features, None, dropout, activation)
        self.fcout = Mlp(features, None, dropout, activation)
        self.attention = MultiHeadAttention(features,
                                            nheads=attention_heads,
                                            scoring_fn=attention_scoring_fn,
                                            heads_reduction=attention_reduction,
                                            layer_idx=layer_idx)
        if filtration:
            self.film = FiltrationBlock(features)
    
    def forward(self, x: th.Tensor, 
                use_cache: bool=False,
                cache_position: Optional[th.LongTensor]=None,
                past_key_values: Optional[Cache]=None):
        x = self.fcin(x)
        x, present_key_values = self.attention(x, 
                        use_cache=use_cache,
                        cache_position=cache_position,
                        past_key_values=past_key_values)
        x = self.fcout(x)
        if hasattr(self, "film"):
            x = self.film(x)
        return BlockOutput(x, present_key_values)

class BlockStack(nn.Module):
    def __init__(self, features: int,
                depth: int=3,
                dropout: float=0.0,
                actiavtion: str="gelu",
                filtration: bool=True,
                n_attn_heads: int=4,
                attention_reduction: str="w-sum",
                attention_scoring_fn: str="scaled-dot-product",
                residual_connections: bool=False,
                last_layer_idx: Optional[int]=None):
        
        super(BlockStack, self).__init__()
        self.blocks: List[nn.Module] = []
        last_layer_idx = (0 if last_layer_idx is None else last_layer_idx)
        for layer_idx in range(depth):
            layer_idx += last_layer_idx
            block = Block(features,
                        dropout,
                        actiavtion,
                        n_attn_heads,
                        attention_reduction,
                        attention_scoring_fn,
                        filtration,
                        layer_idx=layer_idx)
            if residual_connections:
                block = Residual(block)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x: th.Tensor, 
                use_cache: bool=False,
                cache_position: Optional[th.LongTensor]=None,
                past_key_values: Optional[Cache]=None):
        hidden_features = []
        for block in self.blocks:
            bout = block(x, 
                        use_cache=use_cache,
                        cache_position=cache_position,
                        past_key_values=past_key_values)
            (x, past_key_values) = (bout.features, bout.past_key_values)
            hidden_features.append(x)
        return BlockStackOutput(x, 
                                tuple(hidden_features), 
                                past_key_values)


class FusionEnterBlock(nn.Module):
    def __init__(self, text_features: int,
                dynamic_features: int,
                features: int,
                use_mixed_values: bool=True):
        super(FusionEnterBlock, self).__init__()
        linear = lambda f: nn.Sequential(nn.Linear(f, features), nn.LayerNorm(features))
        (self.net_text, self.net_signal) = tuple(linear(f) for f in [text_features, dynamic_features])
        if use_mixed_values:
            self.net_mix = linear(text_features + dynamic_features)
        self.attention = MultiHeadAttention(features, 1, "scaled-dot-product", "w-sum")
    
    def forward(self, text_embeds: TensorType["B", "S", "C"],
                signal_embeds: TensorType["B", "S", "C"],
                mask: Optional[TensorType["B", "S", "S"]]=None):
        x_text = self.net_text(text_embeds)
        x_signal = self.net_signal(signal_embeds)
        x_mix = None
        if hasattr(self, "net_mix"):
            x_mix = th.cat([text_embeds, signal_embeds], dim=-1)
            x_mix = self.net_mix(x_mix)
        attention = self.attention(x_text, x_signal, x_mix, 
                                mask=mask, 
                                use_cache=False)
        return attention
    

if __name__ == "__main__":

    FEATURES = 312
    ATTN_HEADS = 4
    B, S = (10, 256)

    q = th.normal(0, 1, (B, S, FEATURES))
    k = th.normal(0, 1, (B, S, FEATURES))
    v = th.normal(0, 1, (B, S, FEATURES))
    
    block = BlockStack(FEATURES, 
                n_attn_heads=ATTN_HEADS,
                dropout=0.45)
    Nparams = sum([p.numel() for p in block.parameters()])
    x = block(q)
    x = x.hidden_features[-1]
    print(f"Total params count: {Nparams}")
    print(x.shape, x.mean())

        
        



