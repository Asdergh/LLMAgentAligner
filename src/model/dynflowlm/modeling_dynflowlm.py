import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import (Optional, Tuple, Union, Unpack)
from transformers import (AutoConfig, 
                          AutoModel,
                          AutoModelForCausalLM,
                          PreTrainedModel,
                          PretrainedConfig,)
from transformers.cache_utils import Cache
# from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from .configuration_dynflowlm import DynamicFlowLanguageModelConfig
from ..layers import BlockStack, FusionEnterBlock



class DynamicFlowLanguageModel(PreTrainedModel):
    model_type: str = "dynamic-flow-language"
    def __init__(self, config: DynamicFlowLanguageModelConfig):
        super(DynamicFlowLanguageModel, self).__init__(config)
        if config.llm_backbone is not None:
            self.llm_backbone = AutoModelForCausalLM\
            .from_pretrained(config.llm_backbone, 
                             trust_remote_code=True)\
            .eval()
        if config.dyn_backbone is not None:
            self.dyn_backbone = AutoModel\
            .from_pretrained(config.dyn_backbonem, device="cpu")\
            .eval()

        # ===========================Fusion Enter Block===============================
        self.fusion_enter = FusionEnterBlock(text_features=config.tfeatures,
                                             dynamic_features=config.dfeatures,
                                             features=config.hfeatures,
                                             use_mixed_values=config.use_mixed_values)
        # ===========================Fusion Enter Block===============================

        #===========================Fusion Aggregation Blocks=========================
        self.fusion_blocks = nn.ModuleList([BlockStack(features=config.hfeatures,
                                           depth=config.block_depth,
                                           dropout=config.dp,
                                           filtration=config.use_film,
                                           attention_reduction=config.attention_reduction,
                                           attention_scoring_fn=config.attention_score_fn,
                                           residual_connections=config.use_skip)
                                           for _ in range(config.aggregation_depth)])
        #===========================Fusion Aggregation Blocks=========================

    def forward(
        self,
        input_ids: Optional[th.LongTensor] = None,
        attention_mask: Optional[th.Tensor] = None,
        position_ids: Optional[th.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[th.FloatTensor] = None,
        labels: Optional[th.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[th.LongTensor] = None,
        logits_to_keep: Union[int, th.Tensor] = 0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        pass
        


if __name__ == "__main__":
    llm_model = "ai-sage/GigaChat-20B-A3B-instruct"
    config = DynamicFlowLanguageModelConfig(llm_backbone=llm_model)
    model = DynamicFlowLanguageModel(config)
    print(model)