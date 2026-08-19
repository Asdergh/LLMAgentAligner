import torch as th
from dataclasses import (dataclass, field)
from typing import Optional, Literal, List


@dataclass
class ChinaWFBDRecordsDatasetConfig:
    name: str=field(default="physionet-ch", init=False, repr=False, compare=False)
    path: str
    max_subjects: str=100
    time_patch_size: int=100


