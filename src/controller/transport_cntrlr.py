from typing import Tuple

import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Tuple

from matplotlib.transforms import nonsingular

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

    def generate_controls(self, state_extrinsic: Tuple[np.ndarray, ...],
                          target_extrinsic: Tuple[np.ndarray, ...]) -> Tuple[str, np.ndarray]:
        dtype = torch.get_default_dtype()

        # choose a chart free of coordinate singularity

        state_pos_extrinsic = torch.tensor(state_extrinsic[0], dtype=dtype)
        target_pos_extrinsic = torch.tensor(target_extrinsic[0], dtype=dtype)

        nonsingular_chart = self._dynamics.manifold.nonsingular_chart_id(state_pos_extrinsic)

        # computes the state in the intrinsic coordinates
        state_pos_intrinsic = self._dynamics.manifold.to_intrinsic(nonsingular_chart, state_pos_extrinsic)
        state_ts_values = [
            self._dynamics.manifold.to_intrinsic_ts(nonsingular_chart, state_pos_extrinsic,
                                                    torch.tensor(state_ts_val_extrinsic, dtype=dtype)).detach().numpy()
            for state_ts_val_extrinsic in state_extrinsic[1:]
        ]

        # now we will parallel transport the remaining tangent space quantities in the target space back to the tangent
        # space of the current state position

        # NOTE: the tangent space quantities are possibly degenerate if trying to represent the vector in our current
        # chart but because we're parallel transporting it extrinsically then we don't need to worry

        riem_log = self._dynamics.manifold.log(nonsingular_chart, state_pos_intrinsic,
                                               self._dynamics.manifold.to_intrinsic(nonsingular_chart, target_pos_extrinsic)).detach().numpy()

        target_ts_transp_to_state = []

        state_chart = nonsingular_chart
        target_chart = self._dynamics.manifold.nonsingular_chart_id(target_pos_extrinsic)

        for v_target_extrinsic in target_extrinsic[1:]:
            v_target_intrinsic = self._dynamics.manifold.to_intrinsic_ts(target_chart,
                                                                         target_pos_extrinsic,
                                                                         torch.tensor(v_target_extrinsic, dtype=dtype))
            target_ts_transp_to_state.append(
                self._dynamics.manifold.transport_from_q(state_chart, state_pos_intrinsic, target_chart,
                                                         self._dynamics.manifold.to_intrinsic(target_chart, target_pos_extrinsic),
                                                         v_target_intrinsic).detach().numpy())

        controls_state_intrinsic = self.generate_transport_controls(state_chart,
                                                                    tuple([state_pos_intrinsic.detach().numpy(),
                                                                           *state_ts_values]),
                                                                    riem_log,
                                                                    tuple(target_ts_transp_to_state))
        # controls_state_extrinsic = self._dynamics.manifold.to_extrinsic_ts(state_chart,
        #                                                                    state_pos_intrinsic,
        #                                                                    torch.tensor(controls_state_intrinsic,
        #                                                                                 dtype=dtype))
        #
        # # print("TRANSPORT_CNTRLR")
        # # print(f"controls_state_intrinsic: {controls_state_intrinsic}")
        # # print(f"controls_state_extrinsic: {controls_state_extrinsic}")
        #
        # return controls_state_extrinsic.detach().numpy()

        return state_chart, controls_state_intrinsic


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
        state_vel = state[1]
        target_vel = target_ts_transp_to_state[0]

        print("GENERATE TRANSPORT CONTROLS")
        print(f"chart: {chart}")
        print(f"target_vel: {target_vel}")
        print(f"state_vel: {state_vel}")
        print(f"kp gains: {self._kp_gains}")
        print(f"kd gains: {self._kd_gains}")

        print(f"riem_log: {riem_log}")
        print(f"target_vel: {target_vel}")

        controls = self._kp_gains @ riem_log + self._kd_gains @ (target_vel - state_vel)
        print(f"controls: {controls}")
        return controls
