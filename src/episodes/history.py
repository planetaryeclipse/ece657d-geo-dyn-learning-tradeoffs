import numpy as np

from pathlib import Path
from typing import Dict
from dataclasses import dataclass


@dataclass
class IntrinsicHistory:
    pos: np.ndarray
    vel: np.ndarray
    u: np.ndarray
    valid: np.ndarray


@dataclass
class History:
    sample_time: float
    uses_controls: bool

    # extrinsic history
    extrinsic_pos: np.ndarray
    extrinsic_vel: np.ndarray
    extrinsic_u: np.ndarray

    # intrinsic history for all charts
    intrinsic: Dict[str, IntrinsicHistory]

    @staticmethod
    def load(path: Path) -> History:
        data = np.load(path)

        intrinsic = {
            chart: IntrinsicHistory(
                pos=data[f"{chart}_pos"],
                vel=data[f"{chart}_vel"],
                u=data[f"{chart}_u"],
                valid=data[f"{chart}_valid"],
            ) for chart in data['charts']
        }
        return History(
            sample_time=data['sample_time'],
            uses_controls=data['uses_controls'],

            extrinsic_pos=data['extrinsic_pos'],
            extrinsic_vel=data['extrinsic_vel'],
            extrinsic_u=data['extrinsic_u'],

            intrinsic=intrinsic
        )

    def save(self, path: Path):
        history_fields = {
            "sample_time": self.sample_time,
            "uses_controls": self.uses_controls,

            "extrinsic_pos": self.extrinsic_pos,
            "extrinsic_vel": self.extrinsic_vel,
            "extrinsic_u": self.extrinsic_u,

            "charts": list(self.intrinsic.keys())
        }

        for chart, intrinsic_history in self.intrinsic.items():
            history_fields.update({
                f"{chart}_pos": intrinsic_history.pos,
                f"{chart}_vel": intrinsic_history.vel,
                f"{chart}_u": intrinsic_history.u,
                f"{chart}_valid": intrinsic_history.valid
            })

        np.savez(path, **history_fields)
