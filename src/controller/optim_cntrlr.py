import torch
import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

from manifolds.coord_sys import ManifoldCoordSystem


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
    def n(self):
        pass

    @property
    @abstractmethod
    def m(self):
        pass

    @property
    @abstractmethod
    def time(self):
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
    def __init__(self, manifold: ManifoldCoordSystem, initial_state: Tuple[torch.Tensor, torch.Tensor],
                 input_dim: Optional[int] = None,
                 input_dist: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, ):
        super().__init__()
        self._manifold = manifold  # description of curved surface that the state evolves on
        self._initial_state = initial_state

        self._current_time = 0.0
        self._current_state = initial_state

        self._backup_time: Optional[float] = None
        self._backup_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self._input_dim = input_dim if input_dim is not None else manifold.n
        self._numpy_input_dist_fn = (
            lambda _: np.identity(self._manifold.n)  # standard basis of tangent space
            if input_dist is None else
            lambda pos: input_dist(torch.tensor(pos)).numpy())
        self._numpy_christoffel_fn = lambda pos: self._manifold.christoffels(manifold.default_chart,
                                                                             torch.tensor(pos)).detach().numpy()

    @property
    def n(self):
        return self._manifold.n

    @property
    def m(self):
        return self._input_dim

    @property
    def time(self):
        return self._current_time

    @property
    def initial_state(self) -> Tuple[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self._manifold.default_chart, self._initial_state

    @property
    def current_state(self) -> Tuple[str, Tuple[torch.Tensor, torch.Tensor]]:
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


def _compute_step_costs(dynamics: ManifoldPlantDynamics, stacked_horizon_inputs: np.ndarray, dt: float,
                        input_cost: Callable[[float, np.ndarray], float],
                        state_cost: Callable[[float, np.ndarray, np.ndarray], float]) -> float:
    horizon_steps = len(stacked_horizon_inputs) // dynamics.m
    horizon_inputs = stacked_horizon_inputs.reshape((horizon_steps, dynamics.m))

    dynamics.reload_state()

    total_cost = 0.0
    for i in range(horizon_steps):
        current_inputs = horizon_inputs[i, :]
        step_result = dynamics.step(dt, torch.tensor(current_inputs))

        total_cost += input_cost(step_result.time, current_inputs)
        total_cost += state_cost(step_result.time, step_result.pos, step_result.vel)
    return total_cost


class OptimizationController:
    def __init__(self, dynamics: ManifoldPlantDynamics, state_cost: Callable[[float, np.ndarray, np.ndarray], float],
                 input_cost: Callable[[float, np.ndarray], float],
                 horizon_steps: int, horizon_step_dt: float):
        self._dynamics = dynamics
        self._state_cost = state_cost
        self._input_cost = input_cost
        self._horizon_steps = horizon_steps
        self._horizon_step_dt = horizon_step_dt
        self._prev_optimal_horizon_inputs = None

    def generate_optimal_controls(self) -> np.ndarray:
        self._dynamics.save_state()  # system state before advancing during optimization

        guess_horizon_inputs = (
            self._prev_optimal_horizon_inputs if self._prev_optimal_horizon_inputs is not None else np.random.random(
                (self._horizon_steps, self._dynamics.m)))

        step_cost = lambda flat_horizon_inputs: _compute_step_costs(
            self._dynamics, flat_horizon_inputs, self._horizon_step_dt, self._input_cost, self._state_cost)
        result = minimize(step_cost, guess_horizon_inputs.flatten(), method="bfgs")

        print(f"result.x: {result.x}")

        optimal_inputs = result.x.reshape(guess_horizon_inputs.shape)  # optimal inputs at first step

        self._prev_optimal_horizon_inputs = optimal_inputs

        self._dynamics.reload_state()  # restores system state so steps taken during optimization are discarded

        print(f"optimal horizon_inputs: {optimal_inputs}")

        return optimal_inputs[0, :]  # only the next immediate inputs
