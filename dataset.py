import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.typing as npt
from pathlib import Path
from tqdm.auto import tqdm
from scipy.io import loadmat
from typing import List, TypedDict
from torch import Tensor


class RegionMetadataType(TypedDict):
    patch_regions: npt.NDArray[np.uint16]
    weight_decay: npt.NDArray[np.float_]
    source_region: int
    sampled_simulation_idx: int


class MetadataType(TypedDict):
    dataset_len: int
    source_count: int
    nmm_metadata: List[List[RegionMetadataType]]


class EEGDataType(TypedDict):
    eeg: Tensor
    nmm: Tensor
    label: npt.NDArray[np.uint16]


class RawEEGSpikesSingleton(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)

        return cls._instance

    def load(self, spike_path: Path | str, region_count: int = 994):
        viable_data_count = np.array(
            [len([*(spike_path / f"a{i}").iterdir()]) for i in range(region_count)]
        )
        self.spikes = [[] for _ in range(region_count)]

        for i in tqdm(range(region_count)):
            for j in range(1, viable_data_count[i] + 1):
                self.spikes[i].append(
                    loadmat(spike_path / f"a{i}" / f"nmm_{j}.mat")["data"]
                )

    def get(self):
        return self.spikes


class EEGDataset(Dataset):
    def collate_fn(batch):
        return {
            "eeg": torch.stack([x["eeg"] for x in batch]),
            "nmm": torch.stack([x["nmm"] for x in batch]),
            "label": [x["label"] for x in batch],
        }

    def __init__(
        self,
        metadata: MetadataType,
        fwd: npt.NDArray[np.float_],
        duration: int = 500,
        spikes_path: Path | str = Path("source") / "nmm_spikes",
    ):
        self.dataset_len = metadata["dataset_len"]
        self.source_count = metadata["source_count"]
        self.nmm_metadata = metadata["nmm_metadata"]

        self.fwd = fwd
        self.spikes_path = Path(spikes_path)

        self.duration = duration

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, npt.NDArray[np.int16]]:
        raw_nmm = np.zeros((self.duration, self.fwd.shape[1]))
        label = []

        instance = RawEEGSpikesSingleton()

        for patch in self.nmm_metadata[index]:
            label.append(patch["patch_regions"])

            # patch_nmm: npt.NDArray[np.float_] = loadmat(
            #     self.spikes_path
            #     / f"a{patch['source_region']}"
            #     / f"nmm_{patch['sampled_simulation_idx']}.mat"
            # )["data"]

            patch_nmm: npt.NDArray[np.float_] = instance.get()[patch["source_region"]][
                patch["sampled_simulation_idx"] - 1
            ]

            source_signal = patch_nmm[:, [patch["source_region"]]].reshape(-1, 1)

            patch_nmm[:, patch["patch_regions"]] = source_signal * patch["weight_decay"]

            patch_nmm /= patch_nmm.max()

            raw_nmm += patch_nmm

        eeg = raw_nmm @ self.fwd.T
        eeg -= np.mean(eeg, axis=0, keepdims=True)
        eeg -= np.mean(eeg, axis=1, keepdims=True)
        eeg /= np.abs(eeg).max()

        label_cat = np.concatenate(label)
        nmm = np.zeros_like(raw_nmm)
        nmm[:, label_cat] = raw_nmm[:, label_cat]
        nmm /= nmm.max()

        return {
            "eeg": torch.FloatTensor(eeg),
            "nmm": torch.FloatTensor(nmm),
            "label": label,
        }

    def __len__(self):
        return self.dataset_len
