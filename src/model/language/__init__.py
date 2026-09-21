import torch as th
import torch.nn as nn
import lightning as l
from huggingface_hub import repo_exists
from typing import (Optional, Union, Dict, Any, Callable, Literal)
from warnings import warn
from .lm_models import *


class CausalLMGenerationPipeline(l.LightningModule):
    def __init__(self, 
                training_mode: Optional[str]=None,
                student: Union[str, nn.Module, PreTrainedModel]=None,
                teacher: Union[str, nn.Module, PreTrainedModel]=None,
                encoder: Union[str, nn.Module, PreTrainedModel]=None,
                reward_fn: Optional[Callable[...]]=None):
        super(CausalLMGenerationPipeline, self).__init__()
        (self._student, self._student_cfg) = self._get_module(student)
        (self._teacher, self._teacher_cfg) = self._get_module(student)
        (self._reward, _) = self._get_module(reward_fn)

    def _check_training_setup(self):
        """Check current trianing setup configuration"""
        if self._training_mode == "simple":
            if (self._student is not None):
                warn("student model must be initialized \n"
                "for 'simple' training mode!!\n" \
                "You can initilize it using: \n" \
                "   .set_student(...)")
        elif self._training_mode == "distill":
            if (self._student is None)\
                and (self._teacher is None):
                warn("student and teacher model \n"
                "must initialized for 'distil' training mode!! \n" \
                "You can initialize them with: \n" \
                "   .set_student(...)\n" \
                "   .set_teacher(...)\n")
        elif self._training_mode == "finetune":
            if (self._student is None):
                warn("student model must be initialized \n"
                    "for 'simple' training mode!!\n" \
                    "You can initilize it using: \n" \
                    "   .set_student(...)")
            if (self._reward is None):
                warn("reward_fn will unavailable for " \
                    "finetuning process!!!. \n" \
                    "You can initizlie it using: \n" \
                    "   .set_reward(...)")
                
    def set_training_mode(self, mode: Literal["simple", "distill", "finetune"]):
        """set training_mode and check models configuration"""
        self._training_mode = mode
        self._check_training_setup()

    def _get_module(self, source):
        """get module from source"""
        assert (source is not None), ("model source can't be None")
        model = cfg = None
        if isinstance(source, nn.Module):
            model = source
        elif isinstance(source, PreTrainedModel):
            model = source
            cfg = getattr(model, "config", None)
        elif isinstance(source, str):
            if repo_exists(source):
                model = AutoModelForCauslaLM.from_pretrained(source)
                cfg = getattr(model, "config", None)
                cfg = cfg if cfg is not None else AutoModelConfig.from_pretrained(source)
            elif source in _SMALL_CAUSAL_LMS:
                meta = _SMALL_CAUSAL_LMS[source]
                cfg = meta["config"]
                model = meta["model"](config=cfg)
            else:
                raise ValueError(f"Unknown str model source: {source}")
        else:
            raise ValueError(f"unkwno model source type: {type(source)}")
        return (model, cfg)

    def set_student(self, source: str):
        self._student, self._student_cfg = self._get_module(source)
        if hasattr(self, "_reward"):
            pass