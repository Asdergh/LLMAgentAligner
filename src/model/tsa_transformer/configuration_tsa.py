from transformers import PretrainedConfig
from typing import (Optional, Dict, Tuple, Literal, Union)


class TemporalSharedAttentionTransformerConfig(PretrainedConfig):
    model_type = "temporal-shared-attention"
    def __init__(self,
                 channels:            int=1,
                 visual_features:     int=318,
                 patch_size:          Union[int, Tuple[int]]=[14, 32],
                 latent_features:     int=64,
                 latent_act_fn:       str="gelu",
                 visual_act_fn:       str="gelu",
                 aggregation_depth:   int=3,
                 block_depth:         int=2,
                 use_adanorm:         bool=True,
                 time_reduction:      Literal["mean", "sum", "w-sum"]="w-sum",
                 attention_reduction: Literal["mean", "sum", "w-sum"]="w-sum",
                 attention_scoring:   str="scaled-dot-product",
                 attention_heads:     int=4,
                 signal_splits_n:     int=5,
                 skip_connections:    bool=False,
                 **kwargs):

        self.channels = channels
        self.lfeatures = latent_features
        self.vfeatures = visual_features
        self.patch_size = patch_size if isinstance(self.patch_size, tuple) else (patch_size, )*2
        self.aggregation_depth = aggregation_depth
        self.block_depth = block_depth
        self.use_adanorm = use_adanorm
        self.lact_fn = latent_act_fn
        self.vact_fn = visual_act_fn
        self.time_reduction = time_reduction
        self.signal_splits_n = signal_splits_n
        self.attention_reduction = attention_reduction
        self.attention_scoring = attention_scoring
        self.attention_heads = attention_heads
        self.skip = skip_connections
        super(TemporalSharedAttentionTransformerConfig, self).__init__(**kwargs)


