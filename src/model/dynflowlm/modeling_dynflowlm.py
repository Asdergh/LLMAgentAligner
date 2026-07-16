import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, 
                          AutoModel,
                          PreTrainedModel,
                          PretrainedConfig)
from transformers.modeling_utils import ModelOutput
from .configuration_dynflowlm import DynamicFlowLanguageModelConfig
from ..layers import Block, MultiHeadAttention



class DynamicFlowLanguageModel(PreTrainedModel):
    model_type: str = "dynamic-flow-language"
    def __init__(self, config: DynamicFlowLanguageModelConfig):
        super(DynamicFlowLanguageModel, self).__init__(config)
        self.blocks = nn.ModuleList([Block(features=config.hfeatures,
                                           depth=config.block_depth,
                                           dropout=config.dp,
                                           filtration=config.use_film,
                                           attention_reduction=config.attention_reduction,
                                           attention_scoring_fn=config.attention_score_fn,
                                           residual_connections=config.use_skip)
                                           for _ in range(config.aggregation_depth)])
        
        self.sprojection = tuple(nn.Sequential(nn.Linear(f, config.ffeatuers),
                                               nn.GELU(),
                                               nn.LayerNorm(config.ffeatures))
                                               for f in [config.dfeatures, config.sfeatures])
        