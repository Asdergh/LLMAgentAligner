from functools import wraps
from warnings import warn
# from ..data import configuration as mcfg
__PIPEILNES__ = {}
def register_module(name: str):
    def wrapper(cls):
        if name not in __PIPEILNES__:
            __PIPEILNES__[name] = cls
        else:
            raise KeyError(f"pipline {name} is already exists")
        return cls
    return wrapper

def get_module(name: str):
    return __PIPEILNES__[name]