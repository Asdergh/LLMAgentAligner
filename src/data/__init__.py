from .regsitry import __DATASETS__
from .configuration import ChinaWFBDRecordsDatasetConfig

def get_dataset(name):
    return __DATASETS__[name]