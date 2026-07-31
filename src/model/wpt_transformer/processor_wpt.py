import torch as th
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import (savgol_filter, spectrogram)
from skimage.exposure import equalize_hist
from skimage.filters import gaussian
from typing import (Literal, Tuple, Union)
from transformers import FeatureExtractionMixin

class PhysicalExplorationModelProcessor(FeatureExtractionMixin):
    model_input_names = ["spectrograms", "timestampts", "timemask"]
    def __init__(self,
                 normalize: bool=False,
                 normalization: Literal["median", "std", "peak"]="median",
                 smoothe_signal: bool=False,
                 smothing_window_size: int=11,
                 smothing_order: int=3,
                 equalization: bool=True,
                 fft_window: str="hann",
                 nperseg: int=256,
                 noverlap: int=128,
                 size: Union[Tuple[int] | int]=224,
                 gsigma: float=0.0,
                 return_tensors: str="pt",
                 **kwargs):

        self.normalize = normalize
        self.normalization = normalization
        self.smoothe = smoothe_signal
        self.smothing_window_size = smothing_window_size
        self.smothing_order = smothing_order
        self.equalize = equalization
        self.fft_window = fft_window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.size = size
        self.gsigma = gsigma
        self.return_tensors = return_tensors
        super().__init__(**kwargs)

 
    def normalize_signal(self, 
                  signals: np.ndarray, 
                  normtype: str="std"):
        nsignals = signals.copy()
        if normtype == "median":
            nsignals -= np.expand_dims(signals.mean(-1), axis=-1)
        elif normtype == "std":
            mean = np.expand_dims(signals.mean(-1), axis=-1)
            std = np.expand_dims(signals.std(-1), axis=-1)
            nsignals = (signals - mean) / (std + 1e-6)
        elif normtype == "peak":
            mod = np.expand_dims(np.max(np.abs(signals)), axis=-1)
            nsignals /= (mod + 1e-6)
        return nsignals

    def smoothe_signal(self, signals: np.ndarray, 
                       wsize: int=11,
                       order: int=3):
        ChannelsStack = []
        for ch_values in signals:
            sm_values = savgol_filter(ch_values, 
                             window_length=wsize,
                             polyorder=order)
            ChannelsStack.append(sm_values)
        return np.stack(ChannelsStack, axis=0)

    def spectrogram(self, signals: np.ndarray,
                    fs: float,
                    window: str="hann",
                    nperseg: int=256,
                    noverlap: int=128,
                    gsigma: float=0.0,
                    equalize: bool=False,
                    size: int | tuple=224):

        SxxStack = []
        size = size if isinstance(size, tuple) else (size, size)
        values = signals \
            if (signals.ndim != 1) and (signals.shape[0] != 1) \
            else  tqdm(signals,
                       desc="Spectrogram Processing...",
                       ascii=":>",
                       colour="blue")
        for ch_values in values:
            print(nperseg, noverlap)
            print(ch_values.shape)
            (_, _, Sxx) = spectrogram(ch_values,
                                      fs=fs,
                                      window=window,
                                      nperseg=nperseg,
                                      noverlap=noverlap)
            if equalize:
                Sxx = equalize_hist(Sxx)
            Sxx = gaussian(Sxx, sigma=gsigma)
            Sxx = np.array(Image\
                           .fromarray(Sxx)\
                            .resize(size, Image.Resampling.BILINEAR))
            SxxStack.append(Sxx)
        SxxStack = np.stack(SxxStack, axis=-1)
        return SxxStack

    def __call__(self, signals: np.ndarray, fs: float):
        signals = signals \
            if signals.ndim == 2 \
            else np.expand_dims(signals, axis=0)
        print(signals.shape, signals.ndim)
        signals = signals \
            if not self.normalize \
            else self.normalize_signal(signals, self.normalization)
        print(signals.shape)
        signals = signals \
            if not self.smoothe \
            else self.smoothe_signal(signals, 
                                     wsize=self.smothing_window_size,
                                     order=self.smothing_order)
        print(signals.shape)
        spectrograms = self.spectrogram(signals,
                                        fs=fs,
                                        window=self.fft_window,
                                        nperseg=self.nperseg,
                                        noverlap=self.noverlap,
                                        gsigma=self.gsigma,
                                        equalize=self.equalize,
                                        size=self.size)
        output = {"spectrograms": spectrograms,
                  "signals": signals}
        for (k, v) in output.items():
            if isinstance(v, np.ndarray):
                if self.return_tensors == "pt":
                    output[k] = th.from_numpy(v).float()
                elif self.return_tensors == "np":
                    pass
                else:
                    raise ValueError(f"return_tensors field has wrong type: {type(self.return_tensors)}")
        return output


if __name__ == "__main__":

    times = np.linspace(0, 100.6, 1000)
    signals = np.sin(np.pi * times) \
        + np.random.normal(0, 1, (times.shape[0], ))
    fs = 1000
    extractor = PhysicalExplorationModelProcessor(equalization=True,
                                                  nperseg=100,
                                                  noverlap=10,
                                                  normalization="std",
                                                  normalize=True,
                                                  gsigma=1.34)
    output = extractor(signals, fs)
    spec = output["spectrograms"]     
    import matplotlib.pyplot as plt

    print(spec.shape)
    _, axis = plt.subplots()
    axis.imshow(spec.squeeze(), cmap="inferno")
    plt.show()
            
            