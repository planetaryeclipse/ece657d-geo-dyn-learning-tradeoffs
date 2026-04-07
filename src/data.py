from time import sleep

import torch
import numpy as np

from pathlib import Path
from enum import Enum

from torch.utils.data import Dataset

from src.manifolds.sn_mfld import HypersphereManifold
from src.episodes.history import History

from typing import Optional, Tuple, List


class DynamicsDatasetMode(Enum):
    SINGLE_COORD_CHART = 0
    MULTI_COORD_CHART = 1
    MULTI_COORD_CHART_WITH_VALIDITY = 2
    EXTRINSIC_COORD = 3
    SINGLE_COORD_CHART_WITH_EXTRINSIC = 4

    def __str__(self):
        return str(self.value)


def _process_single_chart(history: History, chart: str) -> Tuple[torch.Tensor, torch.Tensor]:
    chart_history = history.intrinsic[chart]
    chart_pos, chart_vel = torch.tensor(chart_history.pos), torch.tensor(chart_history.vel)

    inputs = torch.hstack([chart_pos[:-1, :], chart_vel[:-1, :]])  # current intrinsic position and velocity
    outputs = chart_pos[1:, :]  # updated position

    return inputs, outputs


def _process_multi_chart(history: History, input_chart_order: List[str], output_chart: str) -> Tuple[
    torch.Tensor, torch.Tensor]:
    input_tensors = []  # to be combined with hstack
    output_tensor = None

    for chart in input_chart_order:
        chart_history = history.intrinsic[chart]

        chart_pos, chart_vel = torch.tensor(chart_history.pos), torch.tensor(chart_history.vel)

        input_tensors.extend([chart_pos[:-1, :], chart_vel[:-1, :]])
        if chart == output_chart:
            output_tensor = chart_pos[1:, :]

    inputs = torch.hstack(input_tensors)
    outputs = output_tensor

    return inputs, outputs


def _process_multi_chart_with_validity(history: History, input_chart_order: List[str], output_chart: str) -> Tuple[
    torch.Tensor, torch.Tensor]:
    input_tensors = []  # to be combined with hstack
    output_tensor = None

    for chart in input_chart_order:
        chart_history = history.intrinsic[chart]

        chart_pos, chart_vel = torch.tensor(chart_history.pos), torch.tensor(chart_history.vel)
        chart_validity = torch.tensor(
            np.min(chart_history.valid, axis=1, keepdims=True))  # worst case measure of all intrinsic coords

        input_tensors.extend([chart_pos[:-1, :], chart_vel[:-1, :], chart_validity[:-1, :]])
        if chart == output_chart:
            output_tensor = chart_pos[1:, :]

    inputs = torch.hstack(input_tensors)
    outputs = output_tensor

    return inputs, outputs


def _process_extrinsic(history: History) -> Tuple[torch.Tensor, torch.Tensor]:
    # noinspection DuplicatedCode
    extrinsic_pos, extrinsic_vel = torch.tensor(history.extrinsic_pos), torch.tensor(history.extrinsic_vel)

    inputs = torch.hstack([extrinsic_pos[:-1, :], extrinsic_vel[:-1, :]])  # current extrinsic position and velocity
    outputs = extrinsic_pos[1:, :]  # updated position

    return inputs, outputs


class HypersphereDynamicsDataset(Dataset):
    def __init__(self, history_paths: List[Path], n: int, radius: float, mode: DynamicsDatasetMode,
                 device: torch.Device,
                 output_chart: Optional[str] = None):
        self._n = n
        self._radius = radius
        self._mode = mode

        # gets the chart order
        hs = HypersphereManifold(n, radius)
        self._chart_order = hs.charts
        self._single_chart = self._chart_order[0]

        if output_chart is None:
            output_chart = self._single_chart

        input_tensors = []  # to be combined with vstack
        output_tensors = []  # to be combined with vstack

        for history_path in history_paths:
            history = History.load(history_path)
            match mode:
                case DynamicsDatasetMode.SINGLE_COORD_CHART:
                    history_inputs, history_outputs = _process_single_chart(history, self._single_chart)
                case DynamicsDatasetMode.MULTI_COORD_CHART:
                    history_inputs, history_outputs = _process_multi_chart(history, self._chart_order,
                                                                           self._single_chart)
                case DynamicsDatasetMode.MULTI_COORD_CHART_WITH_VALIDITY:
                    history_inputs, history_outputs = _process_multi_chart_with_validity(history, self._chart_order,
                                                                                         self._single_chart)
                case DynamicsDatasetMode.EXTRINSIC_COORD:
                    history_inputs, history_outputs = _process_extrinsic(history)

            input_tensors.append(history_inputs)
            output_tensors.append(history_outputs)

        # combines all the individual inputs and outputs from all histories into a single tensor which we will access
        # during the training process (note that all these steps have been performed to accelerate this sampling)

        inputs = torch.vstack(input_tensors).to(device)
        outputs = torch.vstack(output_tensors).to(device)

        self._inputs = inputs
        self._outputs = outputs

        self._total_samples = self._inputs.shape[0]

    @staticmethod
    def load(dir_path: Path, n: int, radius: float, mode: DynamicsDatasetMode, device: torch.Device,
             output_chart: Optional[str] = None,
             training_split: float = 0.7) -> Tuple[HypersphereDynamicsDataset, HypersphereDynamicsDataset]:
        # finds all the history path
        history_paths = []
        for child in dir_path.iterdir():
            if child.is_file() and child.name != ".gitkeep":
                history_paths.append(child)

        # performs the training/validation split
        history_paths.sort()  # so training/validation sets remain the same if segmented again

        training_paths = []
        validation_paths = []

        num_training_samples = int(training_split * len(history_paths))
        for i, path in enumerate(history_paths):
            if i < num_training_samples:
                training_paths.append(path)
            else:
                validation_paths.append(path)

        # generates and returns both sets
        training_dataset = HypersphereDynamicsDataset(training_paths, n, radius, mode, device, output_chart)
        validation_dataset = HypersphereDynamicsDataset(validation_paths, n, radius, mode, device, output_chart)

        return training_dataset, validation_dataset

    @property
    def n(self):
        return self._n

    @property
    def radius(self):
        return self._radius

    @property
    def mode(self):
        return self._mode

    @property
    def num_input_features(self):
        return self._inputs.shape[1]

    @property
    def num_output_features(self):
        return self._outputs.shape[1]

    def __len__(self):
        return self._total_samples

    def __getitem__(self, idx):
        return self._inputs[idx, :], self._outputs[idx, :]
