import torch as th
import torch.nn as nn
import torch.nn.functional as Fn
from trnasformers import (PretrianedModel,
                        PretrainedConfig,
                        AutoModel,
                        AutoModelConfig)
from trasnofrmers.modeling_outputs import SequenceClassifierOutput
from .lm_models import *
from ...layers import (get_activation,
                    BlockStack)
from huggingface_hub import repo_exists
from typing import (Optional, Literal)

def create_reward_module(backbone: str):
    if isinstance(backbone, str):
        if repo_exists(backbone):
            model = AutoModel.from_pretrained(backbone)
            config = AutoModelConfig.from_pretrained(backbone)
        elif backbone in _SMALL_CAUSAL_LMS:
            meta = _SMALL_CAUSAL_LMS[backbone]
            (model, config) = (meta["model"], meta["config"])
        else:
            raise ValueError(f"unknown backbone: str source: {backbone}")
    else:
        raise ValueError(f"backbone: str source must str to \n",
                        "Hf repositiry ID/one of available model types" \
                        "You can use list_models() function to get names of all models")


def _reward_config(config_cls: PretrainedConfig):
    
    return Wrapper
        
def _reward_module(model: PretrianedModel):
        
    
    return Wrapper
            


if __name__ == "__main__":
    x = th.normal(0, 1, (10, 3))
    print(x.shape)
    x = Fn.pad(x, (1, 2))
    print(x.shape)
            
            
