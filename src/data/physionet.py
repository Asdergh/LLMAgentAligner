import torch as th
import numpy as np
import abc
import os
import wfdb 
import linecache as lcache
from torch.utils.data import (Dataset, DataLoader)
from .utils import SignalSample
from typing import (Dict, Any, Optional)
from functools import cached_property
from tqdm import tqdm


class PhysioNetDataest(abc.ABC, Dataset):
    def __init__(self, path: str, 
                split_size: int=100):
        super().__init__()
        self.path = path
        self._ss = split_size
        self.data, self.timestampt = self.load_dataset(path)

    @cached_property
    def __len__(self):
        total = 0
        for ssample in self.data.values():
            if isinstance(ssample, SignalSample):
                ts = ssample.values.shape[1]
                n_per_ssample = int(ts // self._ss)
                total += n_per_ssample + (0 if (ts % self.self._ss) == 0 else 1)
        return total

    def __getitem__(self, idx: int):
        assert idx > len(self)
        part_idx = int(idx // len(self))
        return {"signal_values": th.from_numpy(self.data[:, idx*self._ss: (idx + 1)*self._ss]),
                "timestampt": th.from_numpy(self.timestampts[part_idx][idx*self.ss: (idx + 1)*self.ss]),
                "part_idx": part_idx}

    @abc.abstractmethod
    def load_dataset(self, path: str) -> Tuple[dict]:
        raise NotImplemented("you must implement dataset loading function"  \
                            "for every variant os PhysioNetDataset")
    @abc.abstractmethod
    def collate_fn(self, batch: Dict[Any, th.Tensor]):
        raise NotImplemented("you must implement callate fn for every"  \
                            "variant os PhysioNetDataset")


class ChinaWFBDRecordsDataset(PhysioNetDataest):
    def __init__(self, path: str, 
                split_size: int=100,
                max_subjects: int=100):
        self.max_subjects = max_subjects
        super(ChinaWFBDRecordsDataset, self).__init__(path, split_size)
        

    def load_dataset(self, path):
        def load_wfdb(file: str):
            if any([os.path.exists(file + f".{ftype}") 
                    for ftype in ["dat", "hea", "mat"]]):
                data = wfdb.rdrecord(file)
                return SignalSample(values=data.p_signal.T,
                                    sampling_rate=data.fs,
                                    n_channels=data.p_signal.shape[1],
                                    anomaly_ids=data.comments[2]        \
                                                .replace("Dx: ", "")    \
                                                .split(","))
            else:
                return 
        if os.path.exists(path):
            data = {}
            timestampts = {}
            records_txt = os.path.join(path, "RECORDS")
            if not os.path.exists(records_txt):
                raise FileExistsError(("coudn't find RECORDS txt file"      \
                                    f"at location: {records_txt}"           \
                                    "possibly you trying to load dataset"   \
                                    "with wrong data structure"))
            with tqdm(desc="Loading dataset ...",
                    colour="green",
                    total=self.max_subjects) as pbar:
                with open(records_txt, "r") as file:
                    counter_subjects = 0
                    while (counter_subjects <= self.max_subjects):
                        try:
                            line = next(file)
                            rsubfolder = os.path.join(path, line[:-1])
                            rsub_records_txt = os.path.join(rsubfolder, "RECORDS")
                            with open(rsub_records_txt, "r") as rsub_file:
                                for rfile in rsub_file:
                                    rfile =  rfile.replace("\n", "")
                                    rfile_full = os.path.join(rsubfolder, rfile)
                                    ssample = load_wfdb(rfile_full)
                                    if ssample is not None:
                                        data[counter_subjects] = ssample
                                        timestampts[counter_subjects] = np.linspace(0, 1, ssample.tshape)
                                        counter_subjects += 1
                                        pbar.update(1)
                        except StopIteration:
                            break
            return (data, timestampts)

        else:
            raise FileExistsError(f"coudn't find any data at location {path}")

    def collate_fn(self, batch):
        return super().collate_fn(batch)



if __name__ == "__main__":
    path = "/run/media/ramzan/T7/datasets/ch_wfdb_records"
    dataset = ChinaWFBDRecordsDataset(path, 100, 3)
    print(len(dataset.data), dataset.data[0].values.shape)

    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    _, axis = plt.subplots()

    sample = dataset.data[0]
    for idx in range(1, len(dataset.data)):
        if idx == 32:
            break
        sample += dataset.data[idx]
    
    axis.imshow(sample.get_spectrogram(100, 50, size=(50, 224)).mean(axis=0), cmap="jet")
    plt.show()
