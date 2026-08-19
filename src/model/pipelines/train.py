import torch as th
import lightning as l 
import numpy as np
import tyro 
import os
from inspect import signature
from dataclasses import (dataclass, field, is_dataclass, asdict)
from typing import Optional, Dict, Any, Literal, List, Union
from .wpt_model import WPTModel, WPTModelConfig
from ...data import get_dataset, ChinaWFBDRecordsDatasetConfig
from torch.utils.data import DataLoader
from pathlib import Path
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint)
from .registry import get_module


__CALLBACKS__ = {"early_stopping": EarlyStopping, 
                "model_checkpoint": ModelCheckpoint}


from dataclasses import dataclass, field
from typing import Optional, Tuple

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint



@dataclass
class EarlyStoppingConfig:
    monitor: str = "val_loss"
    mode: str = "min"
    min_delta: float = 0.0
    patience: int = 10
    verbose: bool = False
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: bool = False


@dataclass
class ModelCheckpointConfig:
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: Optional[str] = None
    auto_insert_metric_name: bool = True
    every_n_epochs: int = 1
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[Tuple[int, int]] = None
    save_on_train_epoch_end: Optional[bool] = None
    enable_version_counter: bool = True
    verbose: bool = False


@dataclass
class CallbacksConfig:
    early_stopping: bool=True
    model_checkpoint: bool=True
    configurations: Dict[str, object]=field(default_factory=lambda: {
        "early_stopping": asdict(EarlyStoppingConfig()),
        "model_checkpoint": asdict(ModelCheckpointConfig())
    })
    def get_callbcks(self, **kwargs):
        callbacks = []
        for (attr, obj) in signature(self.__init__).parameters.items():
            if not (attr == "configurations"):
                cls = __CALLBACKS__[attr]
                cfg = self.configurations[attr]
                attrs = {k: v for (k, v) in kwargs.items() 
                        if k in signature(cls).parameters}
                cfg.update(attrs)
                cls = cls(**cfg)
                callbacks.append(cls)
        return callbacks
        
    
@dataclass
class TrianingConfig:
    output_dir: Optional[str]=None
    epochs: int=1000
    batch_size: int=64
    initial_lr: float=0.01
    lr_schduler_typ: str="default"
    lr_shceduler_args: Dict[str, Any]=field(default=lambda: {"gamma": 0.8})
    callbacks: CallbacksConfig=field(default_factory=CallbacksConfig)
    monitor: Optional[str]=None
    pipline_config: Union[WPTModelConfig]=field(default_factory=WPTModelConfig)
    logger_config: Dict[str, Any]=field(default_factory=lambda: {"project": "default_lightning"})
    dataset_config: Union[ChinaWFBDRecordsDatasetConfig]=field(default_factory=ChinaWFBDRecordsDatasetConfig)
    dataset_shuffle: bool=False
    dataset_split_parts: Dict[str, float]=field(default_factory=lambda: {
        "train_p": 0.7,
        "val_p": 0.3,
        "test_p": 0.0
    })
    accelerator: str="cuda"

def find_root(path: str | Path):
    path = (path if isinstance(path, Path) else Path(path))
    if not path.exists():
        raise FileExistsError(f"path: {path} is not exists.")

    for root in path.parents:
        if (root / "src").exists() \
            or (root / ".git").exists():
            return str(root)
    return root

def train(config: TrianingConfig):

    dir = config.output_dir
    if dir is None:
        head = find_root(Path(__file__))
        dir = os.path.join(head, "output_dir")
        os.makedirs(dir, exist_ok=True)

    (loaders, min_steps) = {}, float("inf")
    dataset_cfg = (asdict(config.dataset_config) 
                if is_dataclass(config.dataset_config) 
                else config.dataset_config)
    dataset_name = dataset_cfg["name"]
    del dataset_cfg["name"]
    assert isinstance(dataset_cfg, dict), \
    ("dataset_config must be other dataclass instance \n"
    f"or dict. Passed type: {type(dataset_cfg)}.")
    dataset = get_dataset(dataset_name)(**dataset_cfg)\
                    .split_dataset(**config.dataset_split_parts)

    for (loader, dataset) in zip(["train_dataloaders", 
                    "val_dataloaders", 
                    "test_dataloaderss"],
                    list(dataset.values())):
        if dataset is not None:
            loaders[loader] = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=config.dataset_shuffle,
                                collate_fn=dataset.collate_fn)
            min_steps = (min_steps if min_steps < len(dataset) else len(dataset))

    callbacks = config.callbacks\
                .get_callbcks(dirpath=os.path.join(dir, "checkpoints"))
    print(callbacks)
    model = get_module(config.pipline_config.name)(config.pipline_config)
    trainer = l.Trainer(min_epochs=config.epochs,
                        min_steps=min_steps,
                        accelerator=config.accelerator,
                        callbacks=callbacks)
    trainer.fit(model=model,
                **loaders)
        
    


if __name__ == "__main__":
    config = tyro.cli(TrianingConfig)
    train(config)