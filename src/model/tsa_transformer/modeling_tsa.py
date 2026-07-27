import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .configuration_tsa import TemporalSharedAttentionTransformerConfig
from ..layers import *
from ..attention import MultiHeadAttention
from typing import Any

class FlowModel(nn.Module):
    def __init__(self, config: TemporalSharedAttentionTransformerConfig):
        super(FlowModel, self).__init__()
        self.projection = Mlp(features=config.lfeatures, 
                              activation=config.lact_fn)
        if config.use_adanorm:
            self.norm = FiltrationBlock(config.lfeatures)

    def forward(self, t: Any, xt: th.Tensor):
        xt = self.projection(xt)
        xt = xt if not hasattr(self, "norm") else self.norm(xt)
        return xt

class Encoder(nn.Module):
    def __init__(self, config: TemporalSharedAttentionTransformerConfig):
        super(Encoder, self).__init__()
        self.projection = Mlp(features=config.vfeatures,
                              out_features=config.lfeatures,
                              activation=config.lact_fn)
        self.norm = FiltrationBlock(config.lfeatures)
        self.attn = MultiHeadAttention(features=config.lfeatures,
                                       nheads=config.attention_heads,
                                       scoring_fn=config.attention_scoring,
                                       heads_reduction=config.attention_reduction)

    def forward(self, x: th.Tensor):
        x = self.projection(self.projection)
        x = self.norm(x)
        x, _ = self.attn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config: TemporalSharedAttentionTransformerConfig):
        super(Decoder, self).__init__()
        self.projection = Mlp(features=config.lfeatures,
                                      out_features=config.vfeatures,
                                      activation=config.lact_fn)
        self.norm = FiltrationBlock(config.vact_fn)

    def forward(self, x: th.Tensor):
        x = self.projection(x)
        x = self.norm(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, config: TemporalSharedAttentionTransformerConfig):
        super(PatchEmbedding, self).__init__()
        self.cfg = config
        self.conv = nn.Conv2d(config.channels, 
                              config.vfeatures,
                              config.patch_size,
                              config.patch_size,
                              (1, 1))
        if config.use_adanorm:
            self.norm = FiltrationBlock(config.vfeatures)

    def forward(self, image: TensorType["B", "C", "W", "H"]):
        (B, _, W, H) = image.shape
        assert (image.size(-1) % self.cfg.patch_size[-1] == 0),\
        ("wrong image_size: {image.shape[-2:]}"
        "must be dividble by patch_size: {self.cfg.patch_size}")
        embeddings = self.conv(image)
        embeddings = embeddings.view(B, (W // self.cfg.patch_size[0]) 
                                     * (H // self.cfg.patch_size[1]), -1)
        embeddings = embeddings if not hasattr(self, "norm") else self.norm(embeddings)
        return embeddings

class VisualTransformer(nn.Module):
    def __init__(self, config: TemporalSharedAttentionTransformerConfig):
        super(VisualTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList([
            BlockStack(config.vfeatures,
                       config.block_depth,
                       config.vact_fn,
                       config.use_adanorm,
                       config.attention_heads,
                       config.attention_reduction,
                       config.attention_scoring,
                       config.skip)
            for _ in range(config.aggregation_depth)
        ])

    

        
