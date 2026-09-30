import torch as th 
import torch.nn.functional as F
from torchaudio.transforms import (Spectrogram)



def chunk_wave(wave: th.FloatTensor,
                chunk_size: int):
    """raw wave chunking function. """
    n = wave.shape[-1] // chunk_size
    ct_size = wave.shape[-1] % chunk_size
    if ct_size != 0:
        chunk_tail = wave[..., n * chunk_size: ]
        pad_order = (0, chunk_size - ct_size) \
                    if wave.ndim == 1 \
                    else (0, 0, 0, chunk_size - ct_size)
        chunk_tail = th.pad(chunk_tail, pad_order)

    chunks = []
    for idx in range(n):
        chunks.append(wave[..., idx*chunk_size: (idx + 1)*chunk_size])
    if ct_size:
        chunks.append(chunk_tail)
    return th.stack(chunks, dim=-2)
