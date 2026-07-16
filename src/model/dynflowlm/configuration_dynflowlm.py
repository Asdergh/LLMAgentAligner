import transformers as tfs
from transformers import PretrainedConfig
from typing import (Optional)


class DynamicFlowLanguageModelConfig(PretrainedConfig):
    model_type="dynamic_fusser"
    def __init__(self, 
                 dynamic_features:       int=312,
                 semantic_features:      int=312,
                 fusion_features:        int=728,
                 hidden_features:        int=312,
                 block_depth:            int=3,
                 aggregation_depth:      int=4,
                 use_film:               bool=True,
                 attention_nheads:       int=4,
                 use_skip:               bool=False,
                 dropout:                float=0.45,
                 attention_score_fn:     str="softmax",
                 attention_reduction:    str="w-sum",
                 hidden_activation_fn:   str="relu",
                 adaptive_normalization: bool=True,
                 temporal_features:      int=64,
                 dyn_share_fn:           str="default",
                 vocab_size:             int=None,
                 **kwargs):
        
        self.dfeatures = dynamic_features
        self.sfeatures = semantic_features
        self.ffeatuers = fusion_features
        self.hfeatures = hidden_features
        self.block_depth = block_depth
        self.dp = dropout
        self.use_film = use_film
        self.attention_reduction = attention_reduction
        self.attention_score_fn = attention_score_fn
        self.n_attheads = attention_nheads
        self.act_fn = hidden_activation_fn
        self.adanorm = adaptive_normalization
        self.tmp_d = temporal_features
        self.dyn_share_fn = dyn_share_fn
        self.vocab_size = vocab_size
        self.use_skip = use_skip
        self.aggregation_depth = aggregation_depth
        self.llm_backbone = llm_backbone
        self.dyn_backbone = dyn_backbone
        super(DynamicFlowLanguageModelConfig, self).__init__(**kwargs)