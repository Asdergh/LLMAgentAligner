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
from .regsitry import register_dataset

class PhysioNetDataest(abc.ABC, Dataset):
    def __init__(self, path: str, 
                time_patch_size: int=100,
                signal_processing_config: SignalProcessingConfig=None):
        super().__init__()
        self.path = path
        self._tps = time_patch_size
        self.signal_processing_config = signal_processing_config
        self.data: Dict[int, SignalSample] = {}
        self.timestamps: Dict[int, np.ndarray] = {}
        self._pcumsums: list = []

    def __len__(self):
        total = 0
        for ssample in self.data.values():
            if isinstance(ssample, SignalSample):
                ts = ssample.values.shape[1]
                n_per_ssample = int(ts // self._tps)
                total += n_per_ssample + (0 if (ts % self._tps) == 0 else 1)
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

        start_idx = local_idx * self._tps
        last_idx = (local_idx + 1)*self._tps
        if (last_idx > self.data[sample_idx].values.shape[1]):
            last_idx = None

        data = self.data[sample_idx].values[:, start_idx: last_idx]
        times = self.timestamps[sample_idx][start_idx: last_idx]
        (ch, t) = tuple(data.shape)
        if t != self._tps:
            data = np.concatenate([data, np.zeros((ch, self._tps - t))], axis=-1)
            times = np.concatenate([times, np.zeros((self._tps - t, ))])
        return {"waves": th.from_numpy(data),
                "timestamps": th.from_numpy(times),
                "sample_idx": sample_idx}

    @abc.abstractmethod
    def load_dataset(self, path: str) -> None:
        raise NotImplemented("you must implement dataset loading function"  \
                            "for every variant os PhysioNetDataset")
    @abc.abstractmethod
    def collate_fn(self, batch: Dict[Any, th.Tensor]) -> Dict[Any, th.Tensor]:
        raise NotImplemented("you must implement callate fn for every"  \
                            "variant os PhysioNetDataset")

    @abc.abstractmethod
    def split_dataset(self, train_p: float, val_p: float, test_p: float):
        raise NotImplemented("you must implement dataest split fn for every"
                            "variant os PhysioNetDataset")

@register_dataset("physionet-ch", "wpt")
class ChinaWFBDRecordsDataset(PhysioNetDataest):
    def __init__(self, path: str, 
                time_patch_size: int=100,
                max_subjects: int=100):
        super(ChinaWFBDRecordsDataset, self).__init__(path, time_patch_size)
        self.max_subjects = max_subjects
        if path is not None:
            if os.path.exists(path):
                self.load_dataset(path)
            else:
                warn("passing non existing path woud cause FileExistenceError" \
                    "Be sure that data that you trying to load is under location" \
                    "that you specified!!!. " \
                    f"Current locations provided: {path}")
        
        
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
            self.timestamps = {}
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
                    total=self.max_subjects,
                    ascii=":-") as pbar:
                with open(records_txt, "r") as file:
                    counter_subjects = 0
                    try:
                        line = next(file)
                        rsubfolder = os.path.join(path, line[:-1])
                        rsub_records_txt = os.path.join(rsubfolder, "RECORDS")
                        with open(rsub_records_txt, "r") as rsub_file:
                            for rfile in rsub_file:
                                if counter_subjects == self.max_subjects:
                                    break
                                rfile =  rfile.replace("\n", "")
                                rfile_full = os.path.join(rsubfolder, rfile)
                                ssample = load_wfdb(rfile_full)
                                if ssample is not None:
                                    self.data[counter_subjects] = ssample
                                    self.timestamps[counter_subjects] = np.linspace(0, 1, ssample.tshape)
                                    full_ptotal_n = int(ssample.tshape // self._tps)
                                    full_ptotal_n += (0 if ssample.tshape % self._tps == 0 else 1)
                                    full_ptotal_n += (0 if len(self._pcumsums) == 0 else self._pcumsums[counter_subjects - 1])
                                    self._pcumsums.append(full_ptotal_n)
                                    counter_subjects += 1
                                    pbar.update(1)
                    except Exception as e:
                        warn(f"stop loading due to: {e}")
        else:
            raise FileExistsError(f"coudn't find any self.data at location {path}")

    def collate_fn(self, in_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"waves": [],
                "timestamps": [],
                "labels_categorical": [],
                "spectrograms": []}
        for sample in in_batch:
            sidx = sample["sample_idx"]
            if hasattr(self, "labels2indices"):
                label_categorical = th.zeros((len(self.labels2indices), ))
                mask = th.as_tensor(self.labels_from_sample(sidx))
                label_categorical[mask] = 1.0
                batch["labels_categorical"].append(label_categorical)
            spectrogram = th.from_numpy(self.data[sidx].spectrogram).float()
            batch["waves"].append(sample["waves"])
            batch["spectrograms"].append(spectrogram)
            batch["timestamps"].append(sample["timestamps"])

        for (k, v) in batch.items():
            if isinstance(v[0], th.Tensor):
                batch[k] = th.stack(v, dim=0)
        return batch

    def split_dataset(self, train_p: float, 
                    val_p: float, 
                    test_p: float):
        assert (1.0 - (train_p + val_p + test_p)) < 1e-10
        train_n =   int(len(self.data) * train_p)
        val_n =     int(len(self.data) * val_p)
        test_n =    int(len(self.data) * test_p)
        def get_split(sidx: int, eidx: int):
            def handle_dicts(input: dict):
                keys = list(input.keys())[sidx: eidx]
                keys = list(map(lambda x: x - keys[0], keys))
                values = list(input.values())[sidx:eidx]
                return dict(zip(keys, values))
            
            if sidx != len(self.data):
                data = handle_dicts(self.data)
                timestamps = handle_dicts(self.timestamps)
                pcumsums = self._pcumsums[sidx: eidx]
                if sidx != 0:
                    offset = self._pcumsums[:sidx][0]
                    pcumsums = [p - offset for p in pcumsums]
                dataset = ChinaWFBDRecordsDataset(path=None, 
                                            max_subjects=len(data),
                                            time_patch_size=self._tps)
                dataset.data = data
                dataset._pcumsums = pcumsums
                dataset.timestamps = timestamps
                dataset.labels2indices = self.labels2indices
                dataset.indices2labels = self.indices2labels
                dataset.pd_labes = self.pd_labes
                dataset.labels_from_sample = self.labels_from_sample
                return dataset
            else:
                return 

        return {"train": get_split(0, train_n),
                "valset": get_split(train_n, train_n + val_n),
                "testset": get_split(train_n + val_n, train_n + val_n + test_n)}
        


            
        
