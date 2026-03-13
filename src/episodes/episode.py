import numpy as np

from pathlib import Path
from dataclasses import dataclass

from src.episodes.traj import Trajectory

from typing import Any, Dict


@dataclass
class Episode:
    target_traj: Trajectory
    initial_pos: np.ndarray
    initial_vel: np.ndarray
    params: Dict[str, Any]

    @staticmethod
    def load(path: Path) -> Episode:
        data = np.load(path)

        target_traj = Trajectory(
            time=data['_target_traj_time'],
            extrinsic=data['_target_traj_extrinsic'],
            intrinsic={
                key.removeprefix('_target_traj__intrinsic_'): data[key] for key in data.keys() if
                key.startswith('_target_traj__intrinsic_')
            }
        )
        return Episode(
            target_traj=target_traj,
            initial_pos=data['initial_pos'],
            initial_vel=data['initial_vel'],
            params={
                key.removeprefix('_params_'): data[key] for key in data.keys() if key.startswith('_params_')
            }
        )

    def save(self, path: Path) -> None:
        trajectory_fields = {
            "_target_traj_time": self.target_traj.time,
            "_target_traj_extrinsic": self.target_traj.extrinsic,
            **{f"_target_traj__intrinsic_{key}": value for key, value in self.target_traj.intrinsic.items()}
        }
        arg_fields = {
            f"_params_{key}": value for key, value in self.params.items()
        }

        np.savez(path,
                 initial_pos=self.initial_pos,
                 initial_vel=self.initial_vel,
                 **trajectory_fields,
                 **arg_fields)
