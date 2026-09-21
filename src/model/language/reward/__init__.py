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
    class Wrapper(config_cls):
        def __init__(self, 
                    features_chunk: int=256,
                    activations: Literal["relu", "gelu"]="relu",
                    dropout: float=0.23,
                    chunk_size: int=32,
                    chunk_reduction: Literal["weighted", "mean", "sum"]="weighted",
                    num_labels: int=1):
            super(Wrapper, self).__init__()
            self.features_chunk = features_chunk
            self.activations = activations
            self.dp = dropout
            self.chunk_size = chunk_size
            self.chunk_reduction = chunk_reduction
            self.num_labels = num_labels
    return Wrapper
        
def _reward_module(model: PretrianedModel):
        
    class Wrapper(PretrianedModel):
        def __int__(self, config: PretrainedConfig):
            super(Wrapper, self).__init__()
            self._cs = config.chunk_size
            self._d = config.features_chunk
            self.config = config
            self._backbone: PretrianedModel = model
            self._chunk_projection = nn.Sequential(
                nn.Linear(self.config.featuers, self._d),
                nn.LayerNorm(self._d),
                get_activation(self.cfg.actiavtions)
            )
            self._weights = nn.Sequential(
                nn.Linear(self.cfg.features_chunk, 1),
                get_activation("tanh")
            )
            self._aggregation = BlockStack(features=(self._cs * self._d),
                                        depth=self.cfg.aa_aggregation_depth,
                                        dropout=self.cfg.aa_dp,
                                        activation=self.cfg.aa_actiavtions,
                                        filtration=self.cfg.aa_filtration,
                                        n_attn_heads=self.cfg.aa_n_att_heads,
                                        attention_reduction=self.cfg.aa_attention_reduction,
                                        attention_scoring_fn=self.cfg.aa_attention_scoring_fn)
            self._head = nn.Sequential(
                nn.Linear(self.cfg.features_chunk, self.cfg.num_labels),
                get_activation("tanh")
            )

        def chunk_sequence(self, x: th.Tensor):
            """Chunk Sequential Tensor with tokens."""
            chunks = []
            n = (x.shape[1] / self._cs)
            for idx in range(int(n)):
                chunk = x[:, idx*self._cfg: (idx + 1)*self._cfg, :]
                chunks.append(chunk)
            if (n % 1) != 0:
                chunk = x[:, n*self._cs:, :]
                chunks.append(chunk)

            chunks = th.stack()
                
        def forward(self, 
                    input_ids: th.LongTensor,
                    attention_mask: th.LongTensor=None,
                    **kwargs):

            embeddings = self._backbone(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        **kwargs)


if __name__ == "__main__":
    x = th.normal(0, 1, (10, 3))
    print(x.shape)
    x = Fn.pad(x, (1, 2))
    print(x.shape)
            
            
