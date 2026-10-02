import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Literal, List
from dataclasses import (dataclass, field)
from transformers import (PreTrainedModel,
                        PreTrainedConfig,
                        AutoModel,
                        AutoConfig,
                        AutoModelForCausalLM)
from transformers.modeling_outputs import CausalLMOutput
from functools import cached_property
from ...layers import FiltrationBlock, get_activation
from huggingface_hub import repo_exists

# backbones registry
from .backbone_registry import (get_backbone,
                                BackBoneWrapper)
from ....data import get_dataset
from tqdm import tqdm
from warnings import warn
import logging
import os
from torch.utils.data import DataLoader, Dataset



@dataclass
class FusionTrainerConfig:
    language_core:      str # hf repo_id
    fusion_processing:  Literal["cat", "reduct"]="cat"
    optim:              Literal["adam", "sgd"] = "adam"
    optim_kwargs:       Dict[str, Any]=field(default=dict())
    epochs:             int=1000
    bacth_size:         int=32
    shuffle:            bool=False
    backbones:          List[str] = field(default=["hubert-ecg-base"])
    verbose:            bool=True
    verbose_output:     Optional[str]=None
    trainset_kwargs:    Dict[str, Any]=field(default=dict(name="None"))
    valset_kwargs:      Dict[str, Any]=field(default=dict(name="None"))
    testset_kwargs:     Dict[str, Any]=field(default=dict(name="None"))




class FusionTrainer:
    def __init__(self, config: FusionTrainerConfig):
        """Fusion Trainer Class. """
        self.cfg = config
        self._backbones: List[nn.Module] = [
            get_backbone(name) 
            for name in config.backbones
        ]
        try:
            self._llm = AutoModelForCausalLM.from_pretrained(config.language_core)
        except Exception as e:
            raise ValueError(f"problem with finding HF repository with id: {config.language_core} \n"
                            f"Error msg: {e}. ")

        self.ing_logger = None
        if config.verbose:
            self.ing_logger = logging.getLogger()
            logf = "fusion_training.log"\
                    if config.verbose_output is None \
                    else config.verbose_output
            assert os.path.exists(logf), \
            ("Custom info training logger file \n"
            f"doesnt exsits: {logf}!")
            fhandler = logging.FileHandler()
            self.ing_logger.addHandler(fhandler)

        self._train_loader = self._get_logger(config.trainset_kwargs)
        self._val_loader = self._get_logger(config.valset_kwargs)
        self._test_loader = self._get_logger(config.testset_kwargs)

    def set_trainset(self, 
                    cls: Optional[Dataset]=None, 
                    **kwargs):
        name = kwargs.get("name", None)
        data_instance = cls \
                        if cls is not None \
                        else get_dataset(name, kwargs) if name is not None \
                        else None
        assert (data_instance is None), \
        ("possible combination os arguments is: "
        "   1) cls: Dataset class instance with wave/text info in batch"
        "   2) ")
    def _get_logger(self, **kwargs):
        name = kwargs.pop("name")
        data_instance = get_dataset(name)(**kwargs)
        if data_instance is not None:
            return DataLoader(dataset=data_instance,
                            batch_size=self.cfg.bacth_size,
                            shuffle=self.cfg.shuffle)
        return 

    def _fit(self, mode: Literal["train", "val", "test"]="train"):
        """main training function. """
        loader = getattr(self, f"_{mode}_loader")
        assert (loader is None)

        for _ in tqdm(range(self.cfg.epochs),
                        desc="Fusion Training ...",
                        ascii=":.:"):
            for batch in tqdm(self):
                pass


    