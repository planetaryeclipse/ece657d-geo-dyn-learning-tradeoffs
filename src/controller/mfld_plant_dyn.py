import numpy as np
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple

from src.manifolds.coord_sys import ManifoldCoordSystem


@dataclass
class StepResult:
    time: float
    chart: str  # chart the following pos/vel is measured in
    pos: np.ndarray  # current position on the manifold
    vel: np.ndarray  # current velocity on the manifold


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
    def step(self, dt: float, inputs: torch.Tensor) -> StepResult:
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

    input_vecs = np.tensordot(np.diag(inputs), input_basis, ([1], [0]))
    input_total_vec = input_vecs.sum(axis=0)

    dot_pos = vel
    dot_vel = -np.tensordot(np.tensordot(conn_coeffs, vel, ([2], [0])), vel, ([1], [0]))
    dot_vel += input_total_vec

    dot_y = np.concatenate([dot_pos, dot_vel])

    return dot_y


class ManualManifoldPlantDynamics(ManifoldPlantDynamics):
    def __init__(self, manifold: ManifoldCoordSystem, initial_state: Tuple[np.ndarray, np.ndarray],
                 input_dim: Optional[int] = None,
                 input_dist: Optional[Callable[[np.ndarray], np.ndarray]] = None, ):
        super().__init__()
        self._manifold = manifold  # description of curved surface that the state evolves on
        self._initial_state = initial_state

        self._current_time = 0.0
        self._current_state = initial_state

        self._backup_time: Optional[float] = None
        self._backup_state: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._input_dim = input_dim if input_dim is not None else manifold.n
        self._numpy_input_dist_fn = (
            lambda _: np.identity(self._manifold.n)  # standard basis of tangent space
            if input_dist is None else input_dist)
        self._numpy_christoffel_fn = lambda pos: self._manifold.christoffels(
            manifold.default_chart,
            torch.tensor(pos)).detach().numpy()

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
    def initial_state(self) -> Tuple[str, Tuple[np.ndarray, np.ndarray]]:
        return self._manifold.default_chart, self._initial_state

    @property
    def current_state(self) -> Tuple[str, Tuple[np.ndarray, np.ndarray]]:
        return self._manifold.default_chart, self._current_state

    def save_state(self):
        self._backup_time = self._current_time
        self._backup_state = self._current_state

    def reload_state(self):
        self._current_time = self._backup_time
        self._current_state = self._backup_state

    def step(self, dt: float, inputs: np.ndarray) -> StepResult:
        state_pos_numpy, state_vel_numpy = self._current_state
        initial_y = np.concatenate([state_pos_numpy, state_vel_numpy])

        result = solve_ivp(
            lambda t, y: _geodesic_ivp_fn(t, y, inputs, self._numpy_input_dist_fn, self._numpy_christoffel_fn),
            [0, dt], initial_y)

        upd_state_pos, upd_state_vel = result.y[:self._manifold.n, -1], result.y[self._manifold.n:, -1]

        self._current_time += dt
        self._current_state = upd_state_pos, upd_state_vel

        return StepResult(
            time=self._current_time,
            chart=self._manifold.default_chart,
            pos=upd_state_pos,
            vel=upd_state_vel,
        )

    def teardown(self):
        pass  # no teardown actions needed for this implementation
