import numpy as np
import scipy as sp

import torch

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict

from manifolds.coord_sys import ManifoldCoordSystem


@dataclass
class Trajectory:
    time: np.ndarray
    extrinsic: np.ndarray
    intrinsic: Dict[str, np.ndarray]

    def extrinsic_at_t(self, t: float) -> np.ndarray:
        return sp.interpolate.interp1d(self.time, self.extrinsic, axis=0)(t)

    def intrinsic_at_t(self, chart: str, t: float) -> np.ndarray:
        return sp.interpolate.interp1d(self.time, self.intrinsic[chart], axis=0)(t)


def generate_trajectory(start: Union[np.ndarray, float],
                        waypoint_dist: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]],
                        waypoint_dur_dist: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]],
                        num_waypoints: int,
                        dt: float,
                        r: np.random.Generator,
                        coord_sys: ManifoldCoordSystem,
                        gen_chart: Optional[str] = None,
                        interp=sp.interpolate.CubicSpline) -> Trajectory:
    # generates a randomly distributed set of waypoints at random time durations of traversal between each
    wp_pos = [np.array(start, ndmin=1)]
    wp_time = [np.array(0.0, ndmin=0)]

    wp_dist_mean, wp_dist_std = waypoint_dist
    wp_dur_mean, wp_dur_std = waypoint_dur_dist

    # ensures compatability with later numpy methods as for use in multivariate_normal the mean is required to be a
    # 1-dimensional array whereas the covariance is required to have 2 dimensions to prevent throwing an error
    wp_dist_mean, wp_dist_std = np.array(wp_dist_mean, ndmin=1), np.array(wp_dist_std, ndmin=2)
    wp_dur_mean, wp_dur_std = np.array(wp_dur_mean, ndmin=0), np.array(wp_dur_std, ndmin=1)

    for _ in range(num_waypoints):
        prev_wp = wp_pos[-1]
        prev_wp_time = wp_time[-1]

        wp_pos.append(prev_wp + r.multivariate_normal(wp_dist_mean, wp_dist_std))
        wp_time.append(prev_wp_time + r.normal(wp_dur_mean, wp_dur_std).item())

    # samples a smooth episodes joining all the waypoints at the desired sampling frequency
    wp_pos_numpy = np.array(wp_pos)
    wp_time_numpy = np.array(wp_time)

    sample_coords = []
    sample_times = np.arange(wp_time_numpy[0], wp_time_numpy[-1], dt)

    for i in range(wp_pos_numpy.shape[1]):
        sample_coords.append(
            interp(wp_time_numpy, wp_pos_numpy[:, i])(sample_times)
        )
    sample_coords_numpy = np.array(sample_coords).transpose()  # places index by time along dim 0

    # converts generated intrinsic coordinates into a episodes in extrinsic coordinates and the various charts
    # specified in the provided coordinate system (note that this also takes care of any equivalency class in the
    # intrinsic coordinates conversion to extrinsic and then back to intrinsic coordinates will ensure that a unique
    # set of intrinsic coordinates will be utilized for all points along the episodes)

    if gen_chart is None:
        gen_chart = coord_sys.default_chart

    extrinsic = coord_sys.to_extrinsic_batch(gen_chart, torch.as_tensor(sample_coords_numpy))
    intrinsic = {chart: coord_sys.to_intrinsic_batch(chart, extrinsic).numpy() for chart in coord_sys.charts}

    return Trajectory(sample_times, extrinsic.numpy(), intrinsic)
