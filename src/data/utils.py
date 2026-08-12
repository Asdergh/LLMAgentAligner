import wfdb 
import numpy as np
import torch as th 
import os
from dataclasses import dataclass
from typing import (Optional, List, Any)
from scipy.signal import spectrogram, resample
from skimage.exposure import equalize_hist
from skimage.transform import resize
from warnings import warn
from functools import cached_property



@dataclass
class SignalProcessingConfig:
    nperseg:                int = 100 
    noverlap:               int = 50
    scaling:                str = 'density'
    spectrogram_window:     Any = ('tukey_periodic', 0.25)
    spectrogram_size:       int | tuple=224 


@dataclass
class SignalSample:
    values:         np.ndarray=None
    sampling_rate:  Optional[float]=None
    anomaly_ids:    Optional[List[str]]=None
    n_channels:     Optional[int]=None
    p_config:       Optional[SignalProcessingConfig]=None

    def __post_init__(self):
        if self.p_config is not None:
            if self.check_p_config(self.p_config):
                self.set_p_config(self.p_config)

    @property
    def tshape(self):
        return self.values.shape[1]

    def check_p_config(self, p_config):
        if isinstance(p_config, SignalProcessingConfig):
            return True
        else:
            raise ValueError(f"unknow p_config type: {type(self.p_config)}"
                            "only SignalProcessingConfig config instance can be used"
                            "for p_config field, or None !!! ")
            
    def set_p_config(self, p_config: SignalProcessingConfig):
        if self.check_p_config(p_config):
            self.p_config = p_config
            _ = self.spectrogram
        
        
    def _get_argument_cfg(self, argname: str, default: Optional[Any]=None):
        if (self.p_config is not None):
            if self.check_p_config(self.p_config):
                if hasattr(self.p_config, argname):
                    return getattr(self.p_config)
                else:
                    warn(f"No argument: {argname} in p_config")
                    return default
        else:
            warn("p_config if not set")
            return default

    def get_spectrogram(self, nperseg:  int = 100, 
                        noverlap:       int = 50,
                        scaling:        str = 'density',
                        window:         Any = ('tukey_periodic', 0.25),
                        size:           int | tuple=224):

        nperseg = self._get_argument_cfg("nperseg", nperseg)
        noverlap = self._get_argument_cfg("noverlap", noverlap)
        scaling = self._get_argument_cfg("spectrogram_scaling", scaling)
        window = self._get_argument_cfg("window", window)
        size = self._get_argument_cfg("spectrogram_size", size)
        size = size if isinstance(size, tuple) else (size, size)
        """Calculate spectrogrma from signal values"""
        assert (self.values is not None),\
        ("get_spectrogram can only work"
        "with specified values for 'values' field")
        if self.n_channels != self.values.shape[0]:
            self.n_channels = self.values.shape[0]
        SxxStack = []
        for ch_idx in range(self.n_channels):
            (_, _, Sxx) = spectrogram(self.values[ch_idx, :],
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    window=window,
                                    scaling=scaling)
            
            Sxx = equalize_hist(Sxx)
            Sxx = resize(Sxx, size, anti_aliasing=False)
            SxxStack.append(Sxx)
        SxxStack = np.stack(SxxStack, axis=0)
        return SxxStack

    @cached_property
    def spectrogram(self):
        return self.get_spectrogram()

    def __add__(self, other: 'SignalSample'):
        if not isinstance(other, SignalSample):
            raise NotImplemented("add operator for SignalSample works" \
                                "only with another SignalSample")
        assert (self.n_channels == other.n_channels)
        n_channels = self.n_channels
        ovalues = other.values
        if other.sampling_rate != self.sampling_rate:
            new_size = int(other.values.shape[-1] 
                        * (self.sampling_rate 
                        / other.sampling_rate))
            OvaluesStack = []
            for osample in ovalues:
                osample = resample(osample, new_size)
                OvaluesStack.append(osample)
            ovalues = np.stack(OvaluesStack, axis=0)    
        values = np.concatenate([self.values, ovalues], axis=-1)
        anomaliy_ids = list(set(self.anomaly_ids + other.anomaly_ids))
        return SignalSample(values=values,
                            sampling_rate=self.sampling_rate,
                            anomaly_ids=anomaliy_ids,
                            n_channels=n_channels)

    def __radd__(self, other):
        if not isinstance(other, SignalSample):
            raise NotImplemented("add operator for SignalSample works" \
                                            "only with another SignalSample")
        return self.__add__(other)




