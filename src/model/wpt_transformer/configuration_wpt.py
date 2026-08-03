from transformers import PretrainedConfig
from typing import (Optional, Dict, Tuple, Literal, Union)


class WeightedPerceptualTransferConfig(PretrainedConfig):
    model_type = "temporal-shared-attention"
    def __init__(self,
                ode_solver:          str="default",
                in_channels:         int=1,
                out_channels:        int=1,
                time_chunk_size:     int=32,
                tinterpolation_size: int=100,
                image_size:          Union[int, Tuple[int, int]]=224,
                visual_features:     int=364,
                patch_size:          Union[int, Tuple[int]]=14,
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

        self.ode_solver = ode_solver
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_chunk_size = time_chunk_size
        self.tinterpolation_size = tinterpolation_size
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.lfeatures = latent_features
        self.vfeatures = visual_features
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
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
        super(WeightedPerceptualTransferConfig, self).__init__(**kwargs)


