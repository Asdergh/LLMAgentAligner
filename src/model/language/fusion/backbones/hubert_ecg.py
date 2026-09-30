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


class ChunkedTensor:
    def __init__(self, 
                input: th.Tensor, 
                chunk_size: int, 
                dim: int=1,
                chunk_transform_fn: Optional[Callable]=None):
        dim = dim % input.ndim
        def get_chunk(sidx: int, eidx: None):
            """get correct chunk from input tensor. """
            order = tuple(slice(None) 
                        if (i != dim) 
                        else slice(sidx, eidx) 
                        for i in range(input.ndim))
            return input[order].clone()
        self._stack = []
        self.chunk_transform_fn = chunk_transform_fn
        self.n = input.shape[dim] // chunk_size
        self.chunk_tail_size = input.shape[dim] % chunk_size
        if self.chunk_tail_size != 0:
            chunk_tail = get_chunk(self.n*chunk_size)
            pad_order = [0] * (2 * input.ndim)
            pad_order[2 * (input.ndim - 1 - dim) + 1] = chunk_size - self.chunk_tail_size
            chunk_tail = th.pad(chunk_tail, pad_order)
            if self.chunk_transform_fn is not None:
                chunk_tail = self.chunk_trnasform_fn(chunk_tail)
        for idx in range(self.n):
            chunk = get_chunk(idx*chunk_size, (idx + 1)*chunk_size)
            if self.chunk_transform_fn:
                chunk = self.chunk_transform_fn(chunk)
            self._stack.append(chunk)
        if self.chunk_tail_size:
            self._stack.append(chunk_tail)

    def __len__(self):
        return self.n \
                if self.chunk_tail_size == 0 \
                else self.n + 1
    
    def __getitem__(self, idx: int):
        return self._stack[idx]

    def __iter__(self):
        return iter(self._stack) 
    
def input_processor(inputs: TensorType["B", "C", "T"]):
    def chunk_view_fn(chunk: TensorType["B", "C", "T"]):
        return chunk.flatten(start_dim=-2)
    return ChunkedTensor(inputs, 
                        dim=-1, 
                        chunk_size=500,
                        chunk_transform_fn=chunk_view_fn)

def output_processor(features: Iterable):
    

register_backbone("hubert-ecg-base",
                    embedding_dim=768,
                    repo="Edoardo-Coppola/hubert-ecg-base",
                    preprocessor=input_processor)



