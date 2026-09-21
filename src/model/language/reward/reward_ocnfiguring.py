from transformers import PretrainedModelConfig
from ..lm_models import *
from typing import Literal, Optional


def reward_cls(backbone: str):
    base_cls = _SMALL_CAUSAL_LMS[backbone]["config"]
    class RewardModelConfig(base_cls):
        chunk_size: int=32
        chunk_pad_token: str="<|CHUNK|>"
        chunk_reduction: Literal["weighted", "mean", "sum"]="weighted"
        # noise_scdheduling: Literal["default", ""]
    return RewardModelConfig


RewardModelConfig = reward_cls(backbone="GPT2")
