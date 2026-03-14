from typing import Tuple

import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Tuple

from src.controller.mfld_plant_dyn import ManifoldPlantDynamics
from src.manifolds.coord_sys import ManifoldCoordSystem
from src.manifolds.sn_mfld import HypersphereManifold


def _project_vec_onto_basis(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.tensordot(vec, basis, dims=([0], [0])) / torch.tensordot(basis, basis, dims=([0], [0]))


class TransportController(ABC):
    def __init__(self, dynamics: ManifoldPlantDynamics):
        self._dynamics = dynamics

    @property
    def dynamics(self) -> ManifoldPlantDynamics:
        return self._dynamics

    @abstractmethod
    def generate_transport_controls(self, chart: str, state: Tuple[np.ndarray, ...],
                                    riem_log: np.ndarray,
                                    target_ts_transp_to_state: Tuple[np.ndarray, ...]) -> np.ndarray:
        pass

    def generate_controls(self, chart: str, state: Tuple[np.ndarray, ...],
                          target: Tuple[np.ndarray, ...]) -> np.ndarray:
        dtype = torch.get_default_dtype()

        intrinsic_pos = torch.tensor(state[0], dtype=dtype)
        intrinsic_target_pos = torch.tensor(target[0], dtype=dtype)

        riem_log = self._dynamics.manifold.log(chart, intrinsic_pos, intrinsic_target_pos).detach().numpy()
        target_ts_transp_to_state = []

        for v_target in target[1:]:
            target_ts_transp_to_state.append(
                self._dynamics.manifold.transport_from_q(chart, intrinsic_pos, intrinsic_target_pos,
                                                         torch.tensor(v_target, dtype=dtype)).detach().numpy())
        return self.generate_transport_controls(chart, state, riem_log, tuple(target_ts_transp_to_state))


class TransportPDController(TransportController):
    def __init__(self, dynamics: ManifoldPlantDynamics, kp_gains: np.ndarray, kd_gains: np.ndarray):
        super().__init__(dynamics)
        self._kp_gains = kp_gains
        self._kd_gains = kd_gains

    @property
    def kp_gains(self) -> np.ndarray:
        return self._kp_gains

    @property
    def kd_gains(self) -> np.ndarray:
        return self._kd_gains

    def generate_transport_controls(self, chart: str, state: Tuple[np.ndarray, ...],
                                    riem_log: np.ndarray,
                                    target_ts_transp_to_state: Tuple[np.ndarray, ...]) -> np.ndarray:
        state_vel = state[0]
        target_vel = target_ts_transp_to_state[0]

        return self._kp_gains @ riem_log + self._kd_gains @ (target_vel - state_vel)
