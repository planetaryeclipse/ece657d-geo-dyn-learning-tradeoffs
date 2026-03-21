import numpy as np
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple

from src.manifolds.coord_sys import ManifoldCoordSystem

torch.set_default_dtype(torch.float64)  # necessary otherwise tolerances get fairly large


@dataclass
class StepResult:
    time: float

    chart: str
    pos_intrinsic: np.ndarray
    vel_intrinsic: np.ndarray

    pos_extrinsic: np.ndarray
    vel_extrinsic: np.ndarray


# TODO: implement pybullet-based implementation to collect manifold training data for robotic systems

class ManifoldPlantDynamics(ABC):
    def __init__(self):
        pass

    def __del__(self):
        self.teardown()

    @property
    @abstractmethod
    def manifold(self) -> ManifoldCoordSystem:
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        pass

    @property
    @abstractmethod
    def m(self) -> int:
        pass

    @property
    @abstractmethod
    def time(self) -> float:
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def reload_state(self):
        pass

    @abstractmethod
    def step(self, dt: float, inputs_intrinsic: torch.Tensor) -> StepResult:
        pass

    @abstractmethod
    def teardown(self):
        pass


def _geodesic_ivp_fn(_t: float, y: np.ndarray, inputs: np.ndarray, input_dist: Callable[[np.ndarray], np.ndarray],
                     christoffels: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    n = y.shape[0] // 2
    pos, vel = y[:n], y[n:]

    input_basis = input_dist(pos)
    conn_coeffs = christoffels(pos)

    # print("GEODESIC IVP FN")
    # print(f"conn_coeffs: {conn_coeffs}")

    # print("GEODESIC IVP FN")
    # print(f"inputs: {inputs}")
    # print(f"input_basis: {input_basis}")

    input_vecs = np.tensordot(inputs, input_basis, ([0], [1]))
    input_total_vec = input_vecs.sum(axis=0)

    # print(f"input_vecs: {input_vecs}")
    # print(f"input_total_vec: {input_total_vec}")

    dot_pos = vel
    dot_vel = -np.tensordot(np.tensordot(conn_coeffs, vel, ([2], [0])), vel, ([1], [0]))
    dot_vel += input_total_vec
    # print(f"GEODESIC_IVP_FN")
    # print(f"dot_pos: {dot_pos}, dot_vel: {dot_vel}")

    dot_y = np.concatenate([dot_pos, dot_vel])

    return dot_y


class ManualManifoldPlantDynamics(ManifoldPlantDynamics):
    def __init__(self, manifold: ManifoldCoordSystem, state_intrinsic: Tuple[str, np.ndarray, np.ndarray],
                 input_dim: Optional[int] = None,
                 input_dist: Optional[Callable[[np.ndarray], np.ndarray]] = None, ):
        super().__init__()
        self._manifold = manifold  # description of curved surface that the state evolves on

        # converts the internal state to be extrinsic (to allow handling in multiple charts)
        chart, pos_intrinsic, vel_intrinsic = state_intrinsic
        pos_extrinsic = manifold.to_extrinsic(chart, torch.tensor(pos_intrinsic)).detach().numpy()
        vel_extrinsic = manifold.to_extrinsic_ts(chart,
                                                 torch.tensor(pos_intrinsic),
                                                 torch.tensor(vel_intrinsic)).detach().numpy()

        self._current_state_extrinsic = pos_extrinsic, vel_extrinsic
        self._initial_state_extrinsic = self._current_state_extrinsic

        self._current_time = 0.0

        self._backup_time: Optional[float] = None
        self._backup_state: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._input_dim = input_dim if input_dim is not None else manifold.n
        self._input_dist_numpy = (
            lambda _: np.identity(self._manifold.n)  # standard basis of tangent space
            if input_dist is None else input_dist)

    @property
    def manifold(self) -> ManifoldCoordSystem:
        return self._manifold

    @property
    def n(self) -> int:
        return self._manifold.n

    @property
    def m(self) -> int:
        return self._input_dim

    @property
    def time(self) -> float:
        return self._current_time

    @property
    def initial_state_extrinsic(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._initial_state_extrinsic

    @property
    def current_state_extrinsic(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._current_state_extrinsic

    def save_state(self):
        self._backup_time = self._current_time
        self._backup_state = self._current_state_extrinsic

    def reload_state(self):
        self._current_time = self._backup_time
        self._current_state_extrinsic = self._backup_state

    def step(self, dt: float, inputs_extrinsic: np.ndarray) -> StepResult:
        state_pos_extrinsic_numpy, state_vel_extrinsic_numpy = self._current_state_extrinsic

        dtype = torch.get_default_dtype()  # forces torch types (by default numpy is float64)

        state_pos_extrinsic = torch.tensor(state_pos_extrinsic_numpy, dtype=dtype)
        state_vel_extrinsic = torch.tensor(state_vel_extrinsic_numpy, dtype=dtype)

        # print(f"STEP")
        #
        # print(f"state_pos_extrinsic: {state_pos_extrinsic_numpy}, state_vel_extrinsic: {state_vel_extrinsic_numpy}")
        # print(f"inputs_extrinsic: {inputs_extrinsic}")

        # dynamics chart free of singularities
        nonsingular_chart = self._manifold.nonsingular_chart_id(state_pos_extrinsic)

        state_pos_intrinsic = self._manifold.to_intrinsic(nonsingular_chart,
                                                          state_pos_extrinsic).detach().numpy()
        state_vel_intrinsic = self._manifold.to_intrinsic_ts(nonsingular_chart,
                                                             state_pos_extrinsic,
                                                             state_vel_extrinsic).detach().numpy()
        # print(f"extrinsic controls size: {inputs_extrinsic.shape}")

        inputs_intrinsic = self._manifold.to_intrinsic_ts(nonsingular_chart, state_pos_extrinsic,
                                                          torch.tensor(inputs_extrinsic, dtype=dtype)).detach().numpy()
        # print(f"intrinsic controls size: {inputs_intrinsic.shape}")

        # sets up an ivp problem for use in scipy
        initial_y = np.concatenate([state_pos_intrinsic, state_vel_intrinsic])
        christoffels_numpy = lambda pos: self._manifold.christoffels(nonsingular_chart,
                                                                     torch.tensor(pos, dtype=dtype)).detach().numpy()
        result = solve_ivp(
            lambda t, y: _geodesic_ivp_fn(t, y,
                                          inputs_intrinsic,
                                          self._input_dist_numpy,
                                          christoffels_numpy),
            [0, dt], initial_y, method="Radau", dense_output=True)

        # updates the state from the result of the ivp problem
        upd_state_pos_intrinsic, upd_state_vel_intrinsic = result.y[:self._manifold.n, -1], result.y[
            self._manifold.n:, -1]

        upd_state_pos_extrinsic = self._manifold.to_extrinsic(nonsingular_chart,
                                                              torch.tensor(upd_state_pos_intrinsic,
                                                                           dtype=dtype)).detach().numpy()
        upd_state_vel_extrinsic = self._manifold.to_extrinsic_ts(nonsingular_chart,
                                                                 torch.tensor(upd_state_pos_intrinsic, dtype=dtype),
                                                                 torch.tensor(upd_state_vel_intrinsic,
                                                                              dtype=dtype)).detach().numpy()

        self._current_time += dt
        self._current_state_extrinsic = upd_state_pos_extrinsic, upd_state_vel_extrinsic

        return StepResult(
            time=self._current_time,
            chart=nonsingular_chart,
            pos_intrinsic=upd_state_pos_intrinsic,
            vel_intrinsic=upd_state_vel_intrinsic,

            pos_extrinsic=upd_state_pos_extrinsic,
            vel_extrinsic=upd_state_vel_extrinsic,
        )

    def run_for(self, dt: float, tf: float,
                inputs_extrinsic: Optional[np.ndarray] = None) -> StepResult:
        if inputs_extrinsic is None:
            extrinsic_n = self.n + 1
            inputs_extrinsic = np.zeros((extrinsic_n,))

        num_steps = int(tf / dt)
        result = None
        for i in range(num_steps):
            result = self.step(dt, inputs_extrinsic)

        return result  # returns final step

    def teardown(self):
        pass  # no teardown actions needed for this implementation
