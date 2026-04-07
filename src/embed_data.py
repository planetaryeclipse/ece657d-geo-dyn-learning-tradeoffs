from pathlib import Path
from typing import List, Optional, Tuple

from enum import Enum

import torch
from torch.utils.data import Dataset

from src.manifolds.sn_mfld import HypersphereManifold
from src.episodes.history import History


class EmbeddingTrainingMode(Enum):
    TRAIN_NO_ENC_DEC_STREAM = 0
    TRAIN_EXTRINSIC_ENC_DEC_STREAM = 1
    TRAIN_WITH_VEL_ENC_STREAM = 1
    TRAIN_WOUT_VEL_ENC_STREAM = 2

    def __str__(self):
        return str(self.value)


class HypersphereEmbeddedDynamicsDataset(Dataset):

    def __init__(self, history_paths: List[Path], n: int, radius: float, device: torch.Device,
                 chart: Optional[str]):
        self._n = n
        self._radius = radius

        if chart is not None:
            self._chart = chart
        else:
            hs = HypersphereManifold(n, radius)
            self._chart = hs.default_chart

        # following histories will be combined with
        prev_intrin_pos_hist, prev_intrin_vel_hist = [], []
        curr_intrin_pos_hist, curr_intrin_vel_hist = [], []
        next_intrin_pos_hist, next_intrin_vel_hist = [], []

        prev_extrin_pos_hist, prev_extrin_vel_hist = [], []
        curr_extrin_pos_hist, curr_extrin_vel_hist = [], []
        next_extrin_pos_hist, next_extrin_vel_hist = [], []

        for history_path in history_paths:
            history = History.load(history_path)
            intrin_history = history.intrinsic[self._chart]

            intrin_pos, intrin_vel = torch.tensor(intrin_history.pos), torch.tensor(intrin_history.vel)
            extrin_pos, extrin_vel = torch.tensor(history.extrinsic_pos), torch.tensor(history.extrinsic_vel)

            # sample the histories to generate the previous, current, and future position

            prev_intrin_pos, prev_intrin_vel = intrin_pos[:-2, :], intrin_vel[:-2, :]
            curr_intrin_pos, curr_intrin_vel = intrin_pos[1:-1, :], intrin_vel[1:-1, :]
            next_intrin_pos, next_intrin_vel = intrin_pos[2:, :], intrin_vel[2:, :]

            prev_extrin_pos, prev_extrin_vel = extrin_pos[:-2, :], extrin_vel[:-2, :]
            curr_extrin_pos, curr_extrin_vel = extrin_pos[1:-1, :], extrin_vel[1:-1, :]  # used for training dec-enc
            next_extrin_pos, next_extrin_vel = extrin_pos[2:, :], extrin_vel[2:, :]

            # appends the histories (to be joined later)
            prev_intrin_pos_hist.append(prev_intrin_pos)
            prev_intrin_vel_hist.append(prev_intrin_vel)
            curr_intrin_pos_hist.append(curr_intrin_pos)
            curr_intrin_vel_hist.append(curr_intrin_vel)
            next_intrin_pos_hist.append(next_intrin_pos)
            next_intrin_vel_hist.append(next_intrin_vel)

            prev_extrin_pos_hist.append(prev_extrin_pos)
            prev_extrin_vel_hist.append(prev_extrin_vel)
            curr_extrin_pos_hist.append(curr_extrin_pos)
            curr_extrin_vel_hist.append(curr_extrin_vel)
            next_extrin_pos_hist.append(next_extrin_pos)
            next_extrin_vel_hist.append(next_extrin_vel)

        self._prev_intrin_pos_data = torch.vstack(prev_intrin_pos_hist).to(device)
        self._prev_intrin_vel_data = torch.vstack(prev_intrin_vel_hist).to(device)
        self._curr_intrin_pos_data = torch.vstack(curr_intrin_pos_hist).to(device)
        self._curr_intrin_vel_data = torch.vstack(curr_intrin_vel_hist).to(device)
        self._next_intrin_pos_data = torch.vstack(next_intrin_pos_hist).to(device)
        self._next_intrin_vel_data = torch.vstack(next_intrin_vel_hist).to(device)

        self._prev_extrin_pos_data = torch.vstack(prev_extrin_pos_hist).to(device)
        self._prev_extrin_vel_data = torch.vstack(prev_extrin_vel_hist).to(device)
        self._curr_extrin_pos_data = torch.vstack(curr_extrin_pos_hist).to(device)
        self._curr_extrin_vel_data = torch.vstack(curr_extrin_vel_hist).to(device)
        self._next_extrin_pos_data = torch.vstack(next_extrin_pos_hist).to(device)
        self._next_extrin_vel_data = torch.vstack(next_extrin_vel_hist).to(device)

        self._total_samples = self._prev_intrin_pos_data.shape[0]

    @staticmethod
    def load(dir_path: Path, n: int, radius: float, device: torch.Device, chart: Optional[str] = None,
             training_split: float = 0.7) -> Tuple[
        HypersphereEmbeddedDynamicsDataset, HypersphereEmbeddedDynamicsDataset]:
        # finds all the history paths

        # noinspection DuplicatedCode
        history_paths = []
        for child in dir_path.iterdir():
            if child.is_file() and child.name != ".gitkeep":
                history_paths.append(child)

        # perform the training/validation split
        history_paths.sort()  # so training/validation sets remain the same if segmented again

        training_paths = []
        validation_paths = []

        num_training_samples = int(training_split * len(history_paths))
        for i, path in enumerate(history_paths):
            if i < num_training_samples:
                training_paths.append(path)
            else:
                validation_paths.append(path)

        # generates both sets
        training_dataset = HypersphereEmbeddedDynamicsDataset(training_paths, n, radius, device, chart)
        validation_dataset = HypersphereEmbeddedDynamicsDataset(validation_paths, n, radius, device, chart)

        return training_dataset, validation_dataset

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int):
        return (
            # intrinsic
            self._prev_intrin_pos_data[idx, :],
            self._prev_intrin_vel_data[idx, :],
            self._curr_intrin_pos_data[idx, :],
            self._curr_intrin_vel_data[idx, :],
            self._next_intrin_pos_data[idx, :],
            self._next_intrin_vel_data[idx, :],
            # extrinsic
            self._prev_extrin_pos_data[idx, :],
            self._prev_extrin_vel_data[idx, :],
            self._curr_extrin_pos_data[idx, :],
            self._curr_extrin_vel_data[idx, :],
            self._next_extrin_pos_data[idx, :],
            self._next_extrin_vel_data[idx, :],
        )
