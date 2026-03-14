import torch
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from src.controller.mfld_plant_dyn import ManifoldPlantDynamics


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
