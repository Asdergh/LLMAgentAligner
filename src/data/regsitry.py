from functools import wraps
from warnings import warn
# from ..data import configuration as mcfg
__DATASETS__ = {}
def register_dataset(name: str, pipeline: str):
    def wrapper(cls):
        if name not in __DATASETS__:
            # if not hasattr(mcfg, f"{cls.__name__}Config"):
            #     raise RuntimeError("you tried to register dataclass \n"             \
            #                         "without providing its configuration class \n"  \
            #                         "in data/configuration.py.")
            __DATASETS__[name] = cls
        else:
            raise KeyError(f"dataset {name} is already exists")
        return cls
    return wrapper