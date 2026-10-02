from .regsitry import __DATASETS__
from .configuration import ChinaWFBDRecordsDatasetConfig

def get_dataset(name, **kwargs):
    return __DATASETS__[name](**kwargs)