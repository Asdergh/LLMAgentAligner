import torch as th
import torch.nn as nn
import lightning as l
from huggingface_hub import repo_exists
from typing import (Optional, Union, Dict, Any, Callable, Literal, List)
from warnings import warn
from .lm_models import *
from transformers import (PreTrainedModel,
                        AutoModel,
                        AutoConfig,
                        AutoModelForCausalLM,)
from dataclasses import dataclass, field


@dataclass
class LMPOutput:
    texts: Optional[List[str] | str]=None
    prompt_logits: Optional[th.FloatTensor]=None
    prompt_embeddings: Optional[th.Tensor]=None
    wave_predictions: Optional[th.Tensor]=None
    reward_score: Optional[List[float] | float]=None
    texts_general_descriptor: Optional[th.Tensor]=None
    wave_prompt_embeddings: Optional[th.Tensor]=None

class LanguageModelingPipeline(l.LightningModule):
    def __init__(self, 
                training_mode: Optional[str]=None,
                student: Union[str, nn.Module, PreTrainedModel]=None,
                teacher: Union[str, nn.Module, PreTrainedModel]=None,
                encoder: Union[str, nn.Module, PreTrainedModel]=None,
                reward_fn: Union[str, nn.Module, PreTrainedModel]=None):
        super(LanguageModelingPipeline, self).__init__()
        self.set_student(student)
        self.set_teacher(teacher)
        self.set_reward_fn(reward_fn)

        self._training_mode = training_mode
        self._criterions: Dict[str, nn.Module] = {
            "distill":  None,
            "finetune": None,
            "strict":   None
        }

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
        self.criterion = self._criterions[mode]
        if self.criterion:
            warn(f"Criterion module for {mode} is not initialized \n",
                "You can use set_criterion(module, mode) to initialize \n" \
                "it for father training \n ")


    def _get_module(self, source):
        """get module from source"""
        model = backbone = cfg = tokenizer = None
        if source is not None:
            if isinstance(source, nn.Module):
                model = source
            elif isinstance(source, PreTrainedModel):
                model = source
                cfg = getattr(model, "config", None)
            elif isinstance(source, str):
                if repo_exists(source):
                    model = AutoModelForCausalLM.from_pretrained(source)
                    backbone = AutoModel.from_pretrained(source)
                    cfg = getattr(model, "config", None)
                    cfg = cfg if cfg is not None else AutoConfig.from_pretrained(source)
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(source)
                    except Exception as e:
                        warn(f"coudn't load {source} tokenizer dues to: {e}")
                elif source in _SMALL_CAUSAL_LMS:
                    meta = _SMALL_CAUSAL_LMS[source]
                    cfg = meta["config"]
                    model = meta["model"](config=cfg)
                else:
                    raise ValueError(f"Unknown str model source: {source}")
            else:
                raise ValueError(f"unknown model source type: {type(source)}")
        return (model, backbone, cfg, tokenizer)

    def set_student(self, source: str):
        (
            self._student_lm,
            self._student_backbone, 
            self._student_cfg, 
            self._student_tokenizer
        ) = self._get_module(source)

    def set_teacher(self, source: str):
        (
            self._teacher_lm,
            self._teacher_backbone, 
            self._teacher_cfg, 
            self._teacher_tokenizer
        ) = self._get_module(source)

    def set_reward_fn(self, source: str):
        self._reward, _, _, _ = self._get_module(source)

    def set_encoder(self, source: str):
        (
            self._encoder_head, 
            self._encoder_backbone,
            self._encoder_cfg,
            _, _
        )= self._get_module(source)

    def set_criterion(
        self, 
        value: Callable[..., nn.Module], 
        mode: Literal["distill", "finetune", "fuse"]
    ):
        self._criterions[mode] = value
        if hasattr(self, "criterion") \
            and (self._training_mode is not None):
            if (self.criterion is None) \
                and (mode == self._trianing_mode):
                self.criterion = value

    # def predict_step(self, *args, **kwargs):
    #     return super().predict_step(*args, **kwargs)

    def predict(self, 
                texts: Optional[List[str] | str]=None,
                input_ids: Optional[th.LongTensor]=None,
                wave_prompt: Optional[th.FloatTensor]=None,
                max_new_tokens: int=100):

        
        # (B, S, C) = wave_prompt.shape
        # assert B == (1 if isinstance(texts, str) else len(texts))
        if hasattr(self, "_student_lm")               \
        and (self._student_backbone is not None)      \
        and (self._student_tokenizer is not None):
            # TODO make correct wave_prompt fusion procedure
            txt_encode = self._student_tokenizer(texts, return_tensors="pt")
            embeddings = self._student_backbone(**txt_encode).last_hidden_state
            indices = self._student_lm.generate(**txt_encode, 
                                            max_new_tokens=max_new_tokens)
            text = self._student_tokenizer.decode(indices)
            return LMPOutput(texts=text, prompt_embeddings=embeddings)

    def _distill_step(self, batch, batch_idx):
        #TODO setup distillation step_fn
        pass
        
    def _step(self, batch: Any, batch_idx: int, mode: Literal["train", "test", "val"]):
        #TODO setup distill/strict training _step functions
        pass

    def finetune(self):
        pass

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torchы
    torch.manual_seed(42)

    pipeline = LanguageModelingPipeline(training_mode="strict")
    pipeline.set_student("ai-forever/rugpt3small_based_on_gpt2")
    texts = pipeline.predict(["приет, можешь описать анатомию мозга вблизи таламуса ? "]).texts
    print(texts)
    

    
