from transformers import PretrainedConfig, AutoConfig
from typing import Union
from huggingface_hub import repo_exists
from .lm_models import _SMALL_CAUSAL_LMS

class RewardConfig(PretrainedConfig):
    model_type = "chunked_reward"
    sub_configs = {"backbone_config": AutoConfig}
    def __init__(
        self,
        backbone_config: Union[str, dict, PretrainedConfig, None] = None,
        chunk_size: int = 32,
        chunk_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        aggregation_depth: int = 2,
        aggregation_heads: int = 8,
        aggregation_n_att_heads: int = 4,
        aggregation_residuals: bool=True,
        chunk_reduction: str = "weighted",
        num_outputs: int = 1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(backbone_config, str):
            if repo_exists(backbone_config):
                backbone_config = AutoConfig.from_pretrained(backbone_config)
            elif backbone_config in _SMALL_CAUSAL_LMS:
                backbone_config = _SMALL_CAUSAL_LMS[backbone_config]["config"]
            else:
                raise ValueError(f"unkwno backbone_config: str source: {backbone_config}")
        elif not isinstance(backbone_config, PretrainedConfig):
            raise TypeError(f"Unsupported backbone_config type: {type(backbone_config)}")

        self.backbone_config = backbone_config
        self.chunk_size = chunk_size
        self.chunk_dim = chunk_dim
        self.activation = activation
        self.dropout = dropout
        self.aggregation_depth = aggregation_depth
        self.aggregation_heads = aggregation_heads
        self.aggregation_n_attn_heads = aggregation_n_att_heads
        self.aggregation_residuals = aggregation_residuals
        self.chunk_reduction = chunk_reduction
        self.num_outputs = num_outputs
        self.initializer_range = initializer_range

    @property
    def backbone_hidden_size(self) -> int:
        return self.backbone_config.get_text_config().hidden_size