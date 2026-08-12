import torch as th
import numpy as np
import abc
import os
import wfdb 
import linecache as lcache
import pandas as pd
from torch.utils.data import (Dataset, DataLoader)
from .utils import SignalSample, SignalProcessingConfig
from typing import (Dict, Any, Optional, Tuple, List)
from functools import cached_property
from tqdm import tqdm
from warnings import warn
from pprint import pprint


class PhysioNetDataest(abc.ABC, Dataset):
    def __init__(self, path: str, 
                split_size: int=100,
                signal_processing_config: SignalProcessingConfig=None):
        super().__init__()
        self.path = path
        self._ss = split_size
        self.signal_processing_config = signal_processing_config
        self.data: Dict[int, SignalSample] = {}
        self.timestampts: Dict[int, np.ndarray] = {}
        self._pcumsums: list = []

        if path is not None:
            if os.path.exists(path):
                self.load_dataset(path)
            else:
                warn("passing non existing path woud cause FileExistenceError" \
                    "Be sure that data that you trying to load is under location" \
                    "that you specified!!!. " \
                    f"Current locations provided: {path}")
    def __len__(self):
        total = 0
        for ssample in self.data.values():
            if isinstance(ssample, SignalSample):
                ts = ssample.values.shape[1]
                n_per_ssample = int(ts // self._ss)
                total += n_per_ssample + (0 if (ts % self._ss) == 0 else 1)
        return total

    def _get_sequence_idx(self, global_idx: int):
        sequence_idx = 0
        for i, pcum in enumerate(self._pcumsums):
            if pcum >= global_idx:
                sequence_idx = i
                break
        local_pidx = global_idx - (self._pcumsums[sequence_idx - 1] if sequence_idx > 0 else 0)
        return (local_pidx, sequence_idx)
    
    def __getitem__(self, idx: int):
        if self.data is None:
            raise RuntimeError("trying to get self.data before load_datset() call!")
        assert (idx < len(self))
        (local_idx, sample_idx) = self._get_sequence_idx(idx)
        assert (sample_idx < len(self.data))

        start_idx = local_idx * self._ss
        last_idx = (local_idx + 1)*self._ss
        if (last_idx > self.data[sample_idx].values.shape[1]):
            last_idx = None

        data = self.data[sample_idx].values[:, start_idx: last_idx]
        times = self.timestampts[sample_idx][start_idx: last_idx]
        (ch, t) = tuple(data.shape)
        if t != self._ss:
            data = np.concatenate([data, np.zeros((ch, self._ss - t))], axis=-1)
            times = np.concatenate([times, np.zeros((self._ss - t, ))])
        return {"signal_values": th.from_numpy(data),
                "timestampts": th.from_numpy(times),
                "sample_idx": sample_idx}

    @abc.abstractmethod
    def load_dataset(self, path: str) -> None:
        raise NotImplemented("you must implement dataset loading function"  \
                            "for every variant os PhysioNetDataset")
    @abc.abstractmethod
    def collate_fn(self, batch: Dict[Any, th.Tensor]) -> Dict[Any, th.Tensor]:
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
                                                .split(","),
                                    p_config=self.signal_processing_config)
            else:
                return 
        if os.path.exists(path):
            self.data = {}
            self.timestampts = {}
            self._pcumsums = []
            labels_csv = os.path.join(self.path, "ConditionNames_SNOMED-CT.csv")
            if os.path.exists(labels_csv):
                self.pd_labes = pd.read_csv(labels_csv)
                self.labels2indices = {label: idx 
                                    for (idx, label) 
                                    in enumerate(self.pd_labes["Acronym Name"].to_list())}
                self.indices2labels = dict(zip(list(self.labels2indices.values()), 
                                            list(self.labels2indices.keys())))
                def labels_from_sample(sample_idx: int):
                    snomed_labels = map(int, self.data[sample_idx].anomaly_ids)
                    output = []
                    for sn_label in set(snomed_labels):
                        labels = self.pd_labes[self.pd_labes["Snomed_CT"] == sn_label]["Acronym Name"].to_list()
                        indices = [self.labels2indices[label] for label in labels]
                        output += indices
                    return output
                self.labels_from_sample = labels_from_sample

            records_txt = os.path.join(path, "RECORDS")
            if not os.path.exists(records_txt):
                raise FileExistsError(("coudn't find RECORDS txt file"      \
                                    f"at location: {records_txt}"           \
                                    "possibly you trying to load dataset"   \
                                    "with wrong self.data structure"))
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
                                        self.data[counter_subjects] = ssample
                                        self.timestampts[counter_subjects] = np.linspace(0, 1, ssample.tshape)
                                        full_ptotal_n = int(ssample.tshape // self._ss)
                                        full_ptotal_n += (0 if ssample.tshape % self._ss == 0 else 1)
                                        full_ptotal_n += (0 if len(self._pcumsums) == 0 else self._pcumsums[counter_subjects - 1])
                                        self._pcumsums.append(full_ptotal_n)
                                        counter_subjects += 1
                                        pbar.update(1)
                        except StopIteration:
                            break
        else:
            raise FileExistsError(f"coudn't find any self.data at location {path}")

    def collate_fn(self, in_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"signal_values": [],
                "timestampts": [],
                "labels_categorical": [],
                "spectrograms": []}
        for sample in in_batch:
            sidx = sample["sample_idx"]
            if hasattr(self, "labels2indices"):
                label_categorical = th.zeros((len(self.labels2indices), ))
                mask = th.as_tensor(self.labels_from_sample(sidx))
                label_categorical[mask] = 1.0
            spectrogram = th.from_numpy(self.data[sidx].spectrogram).float()
            batch["signal_values"].append(sample["signal_values"])
            batch["spectrograms"].append(spectrogram)
            batch["labels_categorical"].append(label_categorical)
            batch["timestampts"].append(sample["timestampts"])

        for (k, v) in batch.items():
            if isinstance(v[0], th.Tensor):
                batch[k] = th.stack(v, dim=0)
        return batch
