import numpy as np
import torch
import itertools

from src.trajectory.coord_sys import ManifoldCoordSystem

from typing import List
from torch.autograd.functional import jacobian


# Sn manifold (n-dimensional hypersphere smoothly embedded in Rn+1)

def to_intrinsic(euclid: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    euclid_n = euclid.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1
    intrinsic = torch.zeros((n,))

    if n == 1:
        intrinsic[0] = torch.atan2(euclid[1] / radius, euclid[0] / radius)
    else:
        intrinsic[0] = torch.acos(euclid[0] / radius)
        cum_prod = torch.sin(intrinsic[0])

        for i in range(1, n - 1):
            intrinsic[i] = torch.acos(euclid[i] / radius / cum_prod)
            cum_prod = cum_prod * torch.sin(intrinsic[i])  # re-assigned to prevent autograd error when differentiating
        intrinsic[-1] = torch.atan2(euclid[-2] / radius / cum_prod, euclid[-1] / radius / cum_prod)
    return intrinsic


def to_extrinsic(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    n = intrinsic.shape[0]
    euclid = torch.zeros((n + 1,))

    if n == 1:  # implying extrinsic is at least dim 2
        euclid[0] = radius * torch.cos(intrinsic[0])
        euclid[1] = radius * torch.sin(intrinsic[0])
    else:  # implying extrinsic is at least dim 3
        euclid[0] = radius * torch.cos(intrinsic[0])
        cum_prod = torch.sin(intrinsic[0])
        for i in range(1, n - 1):
            euclid[i] = radius * torch.cos(intrinsic[i]) * cum_prod  # re-assigned
            cum_prod = cum_prod * torch.sin(intrinsic[i])
        euclid[-2] = radius * torch.sin(intrinsic[-1]) * cum_prod
        euclid[-1] = radius * torch.cos(intrinsic[-1]) * cum_prod

    return euclid


def _switch_antipodal_coords(coords: torch.Tensor, switch_coords: List[bool]) -> torch.Tensor:
    continuous_coords = (coords + 2 * torch.pi) - torch.Tensor([torch.pi if switch else 0 for switch in switch_coords])
    switched_coords = torch.tensor(
        [coord if coord <= torch.pi else -(2 * torch.pi - coord) for coord in continuous_coords])

    return switched_coords


def to_other_intrinsic(intrinsic: torch.Tensor) -> torch.Tensor:
    n = intrinsic.shape[0]
    total_charts = 2 ** intrinsic.shape[0]  # antipodal chart for each coordinate

    intrinsic_charts = torch.zeros((total_charts, n))
    for i, antipodal in enumerate(itertools.product([False, True], repeat=n)):
        intrinsic_charts[i, :] = _switch_antipodal_coords(intrinsic, antipodal)

    return intrinsic_charts


def metric(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    coord_jacs = jacobian(lambda p: to_extrinsic(p, radius), intrinsic, create_graph=True)
    g = torch.tensordot(coord_jacs, coord_jacs, dims=([0], [0]))
    return g


def christoffels(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    g = metric(intrinsic, radius)
    g_partials = jacobian(lambda p: metric(p, radius), intrinsic,
                          create_graph=True)  # adds index at end due to partials

    # computes the connection coefficients of the Levi-Civita connection using the metric thereby describing the
    # curvature of the n-dimensional hypersphere in the intrinsic coordinate system
    conn_coeffs = 0.5 * torch.tensordot(g.inverse(), g_partials + torch.transpose(g_partials, 1, 2) - torch.transpose(
        torch.transpose(g_partials, 1, 2), 0, 1), dims=([1], [0]))

    return conn_coeffs


def _generate_antipodal_switch(n: int, antipodal_idx: int) -> List[bool]:
    # unlike earlier where we used the cartesian product as iteration over all the antipodal points, we use a more
    # efficient method to prevent generating a large list unnecessarily and rather just treat the number as binary
    # where a value of 1 indicates using the antipodal coord for that chart

    switch_coords = [(antipodal_idx << i) & 1 == 1 for i in range(n)]
    return switch_coords


class HypersphereManifold(ManifoldCoordSystem):
    def __init__(self, n: int, radius: float = 1.0):
        super().__init__(n, n + 1)

        self._radius = radius

        num_charts = 2 ** n  # due to the antipodal points
        self._chart_labels = [f"U{i}" for i in range(num_charts)]
        self._chart_nums = {label: i for i, label in enumerate(self._chart_labels)}

    @property
    def radius(self):
        return self._radius

    @property
    def default_chart(self) -> str:
        return "U0"

    @property
    def charts(self) -> List[str]:
        return self._chart_labels

    def to_intrinsic(self, chart: str, extrinsic: torch.Tensor) -> torch.Tensor:
        default_intrinsic = to_intrinsic(extrinsic, self._radius)
        intrinsic = self.transform_intrinsic(self.default_chart, default_intrinsic, chart)
        return intrinsic

    def to_extrinsic(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        default_intrinsic = self.transform_intrinsic(chart, intrinsic, self.default_chart)
        extrinsic = to_extrinsic(default_intrinsic, self._radius)
        return extrinsic

    def transform_intrinsic(self, current_chart: str, current_intrinsic: torch.Tensor,
                            target_chart: str) -> torch.Tensor:
        current_antipodal_switch = _generate_antipodal_switch(self.n, self._chart_nums[current_chart])
        target_antipodal_switch = _generate_antipodal_switch(self.n, self._chart_nums[target_chart])

        transform_switch = [current != target for current, target in
                            zip(current_antipodal_switch, target_antipodal_switch)]
        return _switch_antipodal_coords(current_intrinsic, transform_switch)

    def intrinsic_weights(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        # for this manifold the chart does not affect the weighting as there are an equal balance of all the charts so
        # we just need to return the scaled distance from the antipodal point (measured in each chart which is the point
        # where the coordinate crossover occurs)
        n = intrinsic.shape[0]
        return torch.sum(torch.pi - torch.abs(intrinsic) / torch.pi) / n
