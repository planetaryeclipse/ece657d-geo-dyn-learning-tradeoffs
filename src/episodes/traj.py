import numpy as np
import scipy as sp

import torch

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict

from manifolds.coord_sys import ManifoldCoordSystem


def _interp_quantities(quantities: Tuple[np.ndarray, ...], time: np.ndarray, t: float) -> Tuple[np.ndarray, ...]:
    interp_quantities = []
    for quantity in quantities:
        interp_quantities.append(
            sp.interpolate.interp1d(time, quantity, axis=0, bounds_error=False, fill_value=(quantity[0], quantity[-1]))(
                t))
    return tuple(interp_quantities)


@dataclass
class Trajectory:
    time: np.ndarray
    extrinsic: Tuple[np.ndarray, ...]
    intrinsic: Dict[str, Tuple[np.ndarray, ...]]

    def extrinsic_at_t(self, t: float) -> Tuple[np.ndarray, ...]:
        return _interp_quantities(self.extrinsic, self.time, t)

    def intrinsic_at_t(self, chart: str, t: float) -> Tuple[np.ndarray, ...]:
        return _interp_quantities(self.intrinsic[chart], self.time, t)


def generate_trajectory(start: Union[np.ndarray, float],
                        waypoint_dist: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]],
                        waypoint_dur_dist: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]],
                        num_waypoints: int,
                        dt: float,
                        r: np.random.Generator,
                        coord_sys: ManifoldCoordSystem,
                        gen_chart: Optional[str] = None,
                        path_diff_order: int = 1,  # num of path derivatives to include
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

    # samples a smooth episodes joining all the waypoints at the desired sampling frequency with the number of specified
    # derivatives of the path for each component (for use in control algorithms)
    wp_pos_numpy = np.array(wp_pos)
    wp_time_numpy = np.array(wp_time)

    sample_coords = [[] for _ in range(path_diff_order + 1)]
    sample_times = np.arange(wp_time_numpy[0], wp_time_numpy[-1], dt)

    for deriv_order in range(path_diff_order + 1):
        for coord_idx in range(wp_pos_numpy.shape[1]):
            # note that if deriv_order is 0 then this is just the interpolated position
            spline_interp = interp(wp_time_numpy, wp_pos_numpy[:, coord_idx]).derivative(deriv_order)
            sample_coords[deriv_order].append(spline_interp(sample_times))

    sample_coords_numpy = tuple(
        np.array(deriv_coords).transpose()  # places index by time along dim 0
        for deriv_coords in sample_coords
    )

    # converts generated intrinsic coordinates into extrinsic coordinates the intrinsic coordinates on the various
    # charts specified in the coordinate system

    # NOTE: this also takes care of any equivalency class in the intrinsic coordinates as the coordinates produced when
    # performing the conversion to extrinsic and back to the intrinsic coordinates will be unique

    if gen_chart is None:
        gen_chart = coord_sys.default_chart

    dtype = torch.get_default_dtype()

    intrinsic_coords = torch.tensor(sample_coords_numpy[0], dtype=dtype)
    extrinsic_coords = coord_sys.to_extrinsic_batch(gen_chart, intrinsic_coords)

    extrinsic = [extrinsic_coords.detach().numpy()]
    for v_intrinsic in sample_coords_numpy[1:]:
        extrinsic.append(
            coord_sys.to_extrinsic_ts_batch(gen_chart, intrinsic_coords,
                                            torch.tensor(v_intrinsic, dtype=dtype)).detach().numpy())
    extrinsic = tuple(extrinsic)

    intrinsic = dict()
    for chart in coord_sys.charts:
        chart_intrinsic = [coord_sys.to_intrinsic_batch(chart, extrinsic_coords).detach().numpy()]
        for v_extrinsic in extrinsic[1:]:
            chart_intrinsic.append(
                coord_sys.to_intrinsic_ts_batch(chart, extrinsic_coords,
                                                torch.tensor(v_extrinsic, dtype=dtype)).detach().numpy())

        intrinsic.update({chart: tuple(chart_intrinsic)})

    return Trajectory(sample_times, extrinsic, intrinsic)
