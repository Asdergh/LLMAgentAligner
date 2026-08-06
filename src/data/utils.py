import wfdb 
import numpy as np
import torch as th 
import os
from dataclasses import dataclass
from typing import (Optional, List, Any)
from scipy.signal import spectrogram, resample
from skimage.exposure import equalize_hist
from skimage.transform import resize


@dataclass
class SignalSample:
    values: np.ndarray=None
    sampling_rate: Optional[float]=None
    anomaly_ids: Optional[List[str]]=None
    n_channels: Optional[int]=None

    @property
    def tshape(self):
        return self.values.shape[1]
    
    def get_spectrogram(self, nperset:  int = 100, 
                        noverlap:       int = 50,
                        scaling:        str = 'density',
                        window:         Any = ('tukey_periodic', 0.25),
                        size:           int | tuple=224):
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
                                    nperseg=nperset,
                                    noverlap=noverlap,
                                    window=window,
                                    scaling=scaling)
            
            Sxx = equalize_hist(Sxx)
            Sxx = resize(Sxx, size, anti_aliasing=False)
            SxxStack.append(Sxx)
        SxxStack = np.stack(SxxStack, axis=0)
        return SxxStack

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




