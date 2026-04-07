from typing import Tuple

import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Tuple

from matplotlib.transforms import nonsingular

from manifolds.sn_mfld import ZERO_NORM_EPS
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
        return self._dynamics

    @abstractmethod
    def generate_transport_controls(self, chart: str, state: Tuple[np.ndarray, ...],
                                    riem_log: np.ndarray,
                                    target_ts_transp_to_state: Tuple[np.ndarray, ...]) -> np.ndarray:
        pass

    def generate_controls(self, state_extrinsic: Tuple[np.ndarray, ...],
                          target_extrinsic: Tuple[np.ndarray, ...]) -> np.ndarray:
        dtype = torch.get_default_dtype()

        # choose a chart free of coordinate singularity

        state_pos_extrinsic = torch.tensor(state_extrinsic[0], dtype=dtype)
        target_pos_extrinsic = torch.tensor(target_extrinsic[0], dtype=dtype)

        print(f"state_pos_extrinsic: {state_pos_extrinsic}, target_pos_extrinsic: {target_pos_extrinsic}")

        nonsingular_state_chart = self._dynamics.manifold.nonsingular_chart_id(state_pos_extrinsic)
        nonsingular_target_chart = self._dynamics.manifold.nonsingular_chart_id(target_pos_extrinsic)

        # computes the state in the intrinsic coordinates
        state_pos_intrinsic = self._dynamics.manifold.to_intrinsic(nonsingular_state_chart, state_pos_extrinsic)
        state_ts_values = [
            self._dynamics.manifold.to_intrinsic_ts(nonsingular_state_chart, state_pos_extrinsic,
                                                    torch.tensor(state_ts_val_extrinsic, dtype=dtype)).detach().numpy()
            for state_ts_val_extrinsic in state_extrinsic[1:]
        ]

        # computes our notion of position error on the manifold
        target_pos_intrinsic_in_state_chart = self._dynamics.manifold.to_intrinsic(nonsingular_state_chart,
                                                                                   target_pos_extrinsic)
        riem_log = self._dynamics.manifold.log(nonsingular_state_chart, state_pos_intrinsic,
                                               target_pos_intrinsic_in_state_chart).detach().numpy()

        target_pos_intrinsic_in_target_chart = self._dynamics.manifold.to_intrinsic(nonsingular_target_chart,
                                                                                    target_pos_extrinsic)
        target_ts_transp_to_state = []
        for target_ts_val_extrinsic in target_extrinsic[1:]:
            target_ts_intrinsic = self._dynamics.manifold.to_intrinsic_ts(
                nonsingular_target_chart, target_pos_extrinsic,
                torch.tensor(target_ts_val_extrinsic, dtype=dtype))
            target_ts_transp_intrinsic = self._dynamics.manifold.transport_from_q(nonsingular_state_chart,
                                                                                  state_pos_intrinsic,
                                                                                  nonsingular_target_chart,
                                                                                  target_pos_intrinsic_in_target_chart,
                                                                                  target_ts_intrinsic).detach().numpy()
            target_ts_transp_to_state.append(target_ts_transp_intrinsic)

        print(
            f"state_pos_intrinsic: {state_pos_intrinsic}, target_pos_intrinsic: {target_pos_intrinsic_in_state_chart}")

        controls_state_intrinsic = self.generate_transport_controls(nonsingular_state_chart,
                                                                    tuple([state_pos_intrinsic.detach().numpy(),
                                                                           *state_ts_values]),
                                                                    riem_log,
                                                                    tuple(target_ts_transp_to_state))
        controls_state_extrinsic = self._dynamics.manifold.to_extrinsic_ts(nonsingular_state_chart,
                                                                           state_pos_intrinsic,
                                                                           torch.tensor(controls_state_intrinsic,
                                                                                        dtype=dtype))
        return controls_state_extrinsic.detach().numpy()


class TransportPDController(TransportController):
    def __init__(self, dynamics: ManifoldPlantDynamics, kp_gains: np.ndarray, kd_gains: np.ndarray,
                 fast_gains: Tuple[float, np.ndarray, np.ndarray]):
        super().__init__(dynamics)
        self._kp_gains = kp_gains
        self._kd_gains = kd_gains
        self._fast_gains = fast_gains

    def generate_transport_controls(self, chart: str, state: Tuple[np.ndarray, ...],
                                    riem_log: np.ndarray,
                                    target_ts_transp_to_state: Tuple[np.ndarray, ...]) -> np.ndarray:
        state_pos, state_vel = state
        target_vel = target_ts_transp_to_state[0]

        state_pos_tensor = torch.tensor(state_pos, dtype=torch.get_default_dtype())
        metric = self._dynamics.manifold.metric(chart, state_pos_tensor).detach().numpy()
        christoffels = self._dynamics.manifold.christoffels(chart, state_pos_tensor).detach().numpy()

        # performs feedback linearization to cancel out the natural acceleration of the geodesic and therefore the
        # evolution of the unforced_dynamics behaves like a linear system
        geod_accel = -np.tensordot(np.tensordot(christoffels, state_vel, ([2], [0])), state_vel, ([1], [0]))

        # print("generate_transport_controls...")
        # print(f"geod_accel: {geod_accel}, state_vel: {state_vel}, christoffels: {christoffels}, christ_shape: {christoffels.shape}")

        print(f"target_vel: {target_vel}")

        err_vel = target_vel - state_vel

        kp_gains, kd_gains = self._kp_gains, self._kd_gains
        # threshold = self._fast_gains[0]
        # riem_log_norm = riem_log.T @ metric @ riem_log
        # if riem_log_norm < threshold:
        #     kp_gains, kd_gains = self._fast_gains[1:]
        # else:
        #     kp_gains, kd_gains = self._kp_gains, self._kd_gains



        # err_vel_norm = err_vel.T @ metric @ err_vel

        # riem_log = riem_log if riem_log_norm < ZERO_NORM_EPS else riem_log / riem_log_norm
        # err_vel = err_vel if err_vel_norm < ZERO_NORM_EPS else err_vel / err_vel_norm

        metric_inv = np.linalg.inv(metric)

        controls = metric @ kp_gains @ riem_log + metric @ kd_gains @ err_vel - geod_accel

        # print(f"riem log: {riem_log}")
        # print(f"intrinsic controls: {controls}")

        # print(f"controls: {controls}")
        return controls
