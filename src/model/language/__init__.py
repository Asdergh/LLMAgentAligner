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
                        AutoModelForCausalLM,
                        PreTrainedTokenizer,
                        AutoTokenizer)
from dataclasses import dataclass, field
from .fusion import (get_backbone, BackBoneWrapper)




class WaveDescripter:
    def __init__(self, 
                llm_core:               Optional[str | nn.Module | PreTrainedModel]=None,
                llm_descriptor:    Optional[str | nn.Module | PreTrainedModel]=None,
                backbones:              Optional[List[str, nn.Module]]=None,
                generation_kwargs:      Dict[str, Any]=dict()):
        """
        Description:
            This class is an implementation of ECG wave annotations generation model.
            Module is separated into three componenets:
                1) llm_core - core linguistic model used nfr to form annotation
                2) llm_descriptor - tiny llm model for CausalLMGEneration task
                    that gnerated breaf description of input 12 chanel ECG forms. 
                3) backbones - wave features extraction modules each with learned embeddin 
                    extractor head that projects there last latent representations into 
                    llm_descriptor embedding space. 
        Notes:
            1) llm_descriptor must have the same tokenizer 
                with llm_code model. This is due to distillation learning of 
                tiny model according to logits space of core. """
        self._backbones: List[BackBoneWrapper] = []
        self._tokenizer: PreTrainedTokenizer = None
        self._llm_core: PreTrainedModel = None
        self._llm_descriptor: PreTrainedModel | nn.Module = None
        self._gen_kwargs = generation_kwargs

        if llm_core is not None:
            self.set_llm_core(llm_core)
        if llm_descriptor is not None:
            self.set_llm_descriptor(llm_descriptor)
        if backbones is not None:
            self.set_backbones(backbones)
        

    def _load_llm(self, src):
        """llm attentioned warned. """
        if isinstance(src, str):
            if repo_exists(src):
                try:
                    return AutoModelForCausalLM.from_pretrained(src)
                except Exception as e:
                    raise ValueError(f"coudn't load model from hf due to: {e}")
            else:
                raise ValueError(f"coudn't find hf repo with id: {src}")
        elif isinstance(src, (nn.Module, PreTrainedModel)):
            return src
        else:
            raise TypeError(f"unknown type for model source: {type(src)}. \n" 
                            "possible variants are: str/nn.Module/PreTrainedModel. ")

    def set_backbones(self, backbones: List[str, BackBoneWrapper]):
        """backbones for module inference can be provided as a list 
        that n contains two different types, str/BackBoneWrapper. 
        In case where item is str type module will try to get bakcbone 
        using get_backbone(...) function, and simple determine module to the 
        backbones if type is BackBoneWrapper. """
        if backbones:
            self._backbones = []
            for idx, value in enumerate(backbones):
                if isinstance(value, str):
                    backbone = get_backbone(value)
                elif isinstance(value, BackBoneWrapper):
                    backbone = value
                else:
                    raise TypeError("wrong type for item in backbones \n" \
                                    f"at possition {idx}. \n"
                                    "possible types are: str/BackbonWrapper .")
                self._backbones.append(backbone)
        if self._backbones and not backbone:
            warn("provided backbones list is empty \n" \
                "Previouds version of backbones list will used during inference. ")

    def set_llm_core(self, llm_source: str):
        """core module must be load with transformers 
        from repository with existing PretrainedTokenizer! """
        self._llm_core = self._load_llm(llm_source)
        self._tokenizer = AutoTokenizer.from_pretrained(llm_source)

    def set_llm_descriptor(self, source: str | nn.Module):
        """lm descriptor can be loaded ither as trained module 
        with its weights or taken from Hf is it was saved before 
        WaveDescriptor usage. """
        self._llm_descriptor = self._load_llm(source)

    def embed_text(self, prompts, **kwargs):
        if isinstance(prompts[0], dict):
            encodes = self._tokenizer\
                        .apply_chat_template(prompts, 
                                            return_tensors="pt",
                                            return_dict=True,
                                            **kwargs)
        elif isinstance(prompts[0], str):
            encodes = self._tokenizer(prompts, 
                                    return_tensors="pt",
                                    **kwargs)
        else:
            raise ValueError(f"prompts elements has wrong type: {type(prompts[0])}"
                            "possible variants of text prompts buffer can be seen from official " \
                            "hf sites and documentation. ")
        embeds = {}
        embeds["core"] = self._llm_core.get_input_embeddings()(encodes["input_ids"])
        if self._llm_descriptor is not None:
            embeds["descriptor"] = self._llm_descriptor.get_input_embedding()(encodes["input_ids"])
        return embeds
        
    def __call__(self,
                prompts: Optional[List[str] | List[Dict[str, str]]]=None,
                waves: Optional[th.FloatTensor]=None,
                **gen_kwargs):

        if (self._tokenizer is None):
            raise RuntimeError("__call__ was called before llm)_core initialization." \
                            "call set_llm_core(...) first to proceed. ")
        embeddings = self.embed_text(prompts)
        