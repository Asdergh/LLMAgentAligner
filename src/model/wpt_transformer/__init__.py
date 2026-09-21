from .modeling_wpt import WPTOutput
from .modeling_wpt import WeightedPerceptualTransferModel
from .modeling_wpt import WPTCriterionModel
from .configuration_wpt import WeightedPerceptualTransferConfig



import torch as th
import numpy as np
import wandb as wdb
import lightning as l
from inspect import signature
from dataclasses import (dataclass, field, asdict, is_dataclass)
from transformers import AutoModel
from typing import Literal, Optional
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from typing import Dict, Any, List
from torchtyping import TensorType
from .utils import create_optimizer
from .registry import register_module


from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class WPTConfigContainer:
    """Dataclass version of WeightedPerceptualTransferConfig."""
    ode_solver: str = "default"
    in_channels: int = 12
    out_channels: int = 12
    time_chunk_size: int = 64
    tinterpolation_size: int = 100
    image_size: Union[int, Tuple[int, int]] = 224
    visual_features: int = 364
    patch_size: Union[int, Tuple[int]] = 14
    latent_features: int = 64
    latent_act_fn: str = "gelu"
    visual_act_fn: str = "gelu"
    aggregation_depth: int = 3
    block_depth: int = 2
    use_adanorm: bool = True
    time_reduction: Literal["mean", "sum", "w-sum"] = "w-sum"
    attention_reduction: Literal["mean", "sum", "w-sum"] = "w-sum"
    attention_scoring: str = "scaled-dot-product"
    attention_heads: int = 4
    signal_splits_n: int = 5
    skip_connections: bool = False
    n_classes: Optional[int] = None

@dataclass
class WPTBaseConfig:
    name: str=field(default="wpt", init=False, repr=False, compare=False)
    model: Optional[str]=None
    config: WPTConfigContainer=field(default_factory=WPTConfigContainer)
    optimizer_type: Literal["sgd", "adam"]="adam"
    optimizer_arguments: Dict[str, Any]=field(default_factory=lambda: dict(type="adam"))
    initial_lr: float=0.01
    gamma: float=0.1

@register_module("wpt")
class WPTBaseModule(l.LightningModule):
    def __init__(self, config: WPTBaseConfig):
        super(WPTBaseModule, self).__init__()
        self.cfg = config
        assert (self.cfg.model is not None) or (self.cfg.config is not None)
        if (self.cfg.config is not None) or (self.cfg.model is None):
            config = (WeightedPerceptualTransferConfig(**asdict(self.cfg.config))
                    if self.cfg.config is not None 
                    else WeightedPerceptualTransferConfig())


            assert isinstance(config, WeightedPerceptualTransferConfig), \
            (f"Wrong model config type: {type(config)}"
            "Expected WeightedPerceptualTransferConfig class instance.")
            self.wpt = WeightedPerceptualTransferModel(config)
            self.wpt_criterion = WPTCriterionModel(config)
        if self.cfg.model is not None:
            self.wpt = AutoModel.from_pretrained(model)
            self.wpt_criterion = WPTCriterionModel(self.wpt.config)

    def configure_optimizers(self):
        arguments = self.cfg.optimizer_params
        name = arguments.pop("type")
        optim = dict(adam=Adam, sgd=SGD)[name](**arguments)
        return {"optimizer": optim, 
                "lr_scheduler": {
                    "scheduler": ExponentialLR(optim, gamma=self.cfg.gamma),
                    "interval": "epoch",
                    "frequency": 1
                }}

    def _step(self, batch: Dict[str, th.Tensor], mode: Literal["train", "val"]="train"):
        (Sxx, Svalues, labels, times) = (batch["spectrograms"], 
                                    batch["waves"], 
                                    batch["labels_categorical"],
                                    batch["timestamps"])
        print(Sxx.shape)
        wpt_output = self.wpt(Sxx, times)
        losses = self.wpt_criterion(wpt_output, Svalues, labels)
        for (k, v) in losses.items():
            self.log(f"{mode}-{k}", v, on_step=True, on_epoch=False)
        return losses["loss"]
    

    def training_step(self, batch: Dict[str, th.Tensor]):
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, th.Tensor]):
            return self._step(batch, "val")
    
    def _epoch_end(self, mode: Literal["train", "val"]):
        cls_stats = self.wpt_criterion.get_classification_stats(True)
        (tp, tn, fp, fn) = tuple(cls_stats.values())

        eps = 1e-5
        accuracy = (tp + tn) / (eps + tp + tn + fp + fn)
        recall = tp / (eps + tp + fn)
        precision = tp / (eps + tp + fp)
        f1 = (2*precision*recall) / (eps + precision + recall)
        self.log(f"{mode}-accuracy", accuracy, on_step=False, on_epoch=True)
        self.log(f"{mode}-recall", recall, on_step=False, on_epoch=True)
        self.log(f"{mode}-f1", f1, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def forward(self, spectrogram: TensorType["B", "C", "W", "H"],
                timestamps: TensorType["B", "T"] | th.FloatTensor):
        return self.wpt(spectrogram, timestamps)

    def predict(self, spectrogram: TensorType["B", "C", "W", "H"],
                timestamps: th.FloatTensor):
        return self.wpt.approximate(spectrogram, timestamps)

    def interpolate(self, spectrogram: TensorType["B", "C", "W", "H"],
                timestamps: th.FloatTensor,
                steps: int | List[int] | th.LongTensor=100,
                verbose: bool=True):
        return self.wpt.interpolate(spectrogram, timestamps, steps, verbose)


        
    


if __name__ == "__main__":
    config = WeightedPerceptualTransferConfig(in_channels=12, 
                                            n_classes=32,
                                            out_channels=12)
    model = WPTBaseModule(config=config)
    print(type(model.logger))
    data = th.normal(0, 1, (10, 12, 224, 224))
    timestamps = th.linspace(0, 1, 100)
    output = model.interpolate(data, timestamps, steps=10)

    print(output.channels_output.shape, 
        output.temporal_cls_tokens.shape)
    
