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


# NOTE: this intrinsic generation scheme only generates within a single chart and as a consequence fails when it reaches
# a singular point so we can't use this

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


def generate_hs_trajectory(pos_start_ambient: np.ndarray,
                           waypoint_dist: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]],
                           waypoint_travel_time_dist: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]],
                           num_waypoints: int,
                           dt: float,
                           radius: float,
                           coord_sys: ManifoldCoordSystem,
                           rand: np.random.Generator,
                           interp=sp.interpolate.CubicSpline) -> Trajectory:
    # generates a random walk of waypoints in the ambient space
    waypoints_ambient_pos = [np.array(pos_start_ambient, ndmin=1)]
    waypoints_time = [np.array(0.0, ndmin=0)]

    waypoint_dist_mean, waypoint_dist_std = waypoint_dist
    waypoint_travel_time_mean, waypoint_travel_time_std = waypoint_travel_time_dist

    # ensures compatibility with later numpy statistical methods
    waypoint_dist_mean, waypoint_dist_std = np.array(waypoint_dist_mean, ndmin=1), np.array(waypoint_dist_std, ndmin=2)
    waypoint_travel_time_mean, waypoint_travel_time_std = np.array(waypoint_travel_time_mean, ndmin=0), np.array(
        waypoint_travel_time_std, ndmin=1)

    for _ in range(num_waypoints):
        prev_waypoint = waypoints_ambient_pos[-1]
        prev_waypoint_time = waypoints_time[-1]

        # normalize the previous waypoint so the eventual projection onto the hypersphere will exhibit some change and
        # will not just be stuck in one location due to having values from the hypersphere
        prev_waypoint /= np.linalg.norm(prev_waypoint)

        new_waypoint = prev_waypoint + rand.multivariate_normal(waypoint_dist_mean, waypoint_dist_std)
        new_waypoint_time = prev_waypoint_time + rand.normal(waypoint_travel_time_mean, waypoint_travel_time_std).item()

        waypoints_ambient_pos.append(new_waypoint)
        waypoints_time.append(new_waypoint_time)

    # samples a smooth trajectory in this ambient space and then projects them only the hypersphere
    waypoints_ambient_pos = np.array(waypoints_ambient_pos)
    waypoints_time = np.array(waypoints_time)

    sample_coords_ambient = [[] for _ in range(2)]  # space for pos and vel
    sample_coords_time = np.arange(waypoints_time[0], waypoints_time[-1], dt)

    for coord_idx in range(waypoints_ambient_pos.shape[1]):
        spline_interp = interp(waypoints_time, waypoints_ambient_pos[:, coord_idx])

        # NOTE: the result from spline interpolation is a row vector with samples of each coordinate along the row (so
        # indexed at the various column positions)
        sample_coords_ambient[0].append(spline_interp(sample_coords_time))
        sample_coords_ambient[1].append(spline_interp.derivative(1)(sample_coords_time))

    # converts the nested list of the samples of each coordinate into a single array with all coordinates and then
    # switches the indexing so that different samples are found at different rows
    sample_coords_ambient_numpy = tuple(
        np.array(deriv_coords).transpose()  # places index by time along dim 0
        for deriv_coords in sample_coords_ambient
    )

    # projects the trajectory onto the hypersphere
    x, dot_x = sample_coords_ambient_numpy

    print(f"x: {x.shape}, dot_x: {dot_x.shape}")
    print(f"norm: {np.linalg.norm(x, axis=1, keepdims=True).shape}")

    hypersphere_x: np.ndarray = radius * x / np.linalg.vector_norm(x, axis=1, keepdims=True)
    hypersphere_x_term_1 = radius * dot_x / np.linalg.vector_norm(x, axis=1, keepdims=True)
    hypersphere_x_term_2 = radius * (
            - np.sum(x * dot_x, axis=1, keepdims=True) / np.linalg.vector_norm(x, axis=1, keepdims=True) ** 3 * x)

    print(f"hypersphere_x_term_1: {hypersphere_x_term_1.shape}")
    print(f"hypersphere_x_term_2: {hypersphere_x_term_2.shape}")

    hypersphere_dot_x: np.ndarray = hypersphere_x_term_1 + hypersphere_x_term_2

    # sets up the tuples to pass into the trajectory dataclass
    extrinsic = (hypersphere_x, hypersphere_dot_x)

    intrinsic = dict()
    for chart in coord_sys.charts:
        intrinsic.update({chart: (coord_sys.to_intrinsic_batch(chart, torch.tensor(hypersphere_x)).detach().numpy(),
                                  coord_sys.to_intrinsic_ts_batch(chart,
                                                                  torch.tensor(hypersphere_x),
                                                                  torch.tensor(hypersphere_dot_x)).detach().numpy())})

    return Trajectory(
        time=sample_coords_time,
        extrinsic=extrinsic,
        intrinsic=intrinsic
    )
