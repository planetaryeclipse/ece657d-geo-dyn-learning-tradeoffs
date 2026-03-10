# Sn manifold (n-dimensional hypersphere smoothly embedded in Rn+1)
from math import prod

import torch

from torch.func import grad
from torch.autograd.functional import jacobian

import itertools

from typing import List


def to_intrinsic(euclid: torch.Tensor) -> torch.Tensor:
    euclid_n = euclid.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1

    intrinsic = torch.zeros((n,))

    if n == 1:
        intrinsic[0] = torch.atan2(euclid[1], euclid[0])
    else:
        intrinsic[0] = torch.acos(euclid[0])
        cum_prod = torch.sin(intrinsic[0])

        for i in range(1, n - 1):
            intrinsic[i] = torch.acos(euclid[i] / cum_prod)
            cum_prod = cum_prod * torch.sin(intrinsic[i])  # re-assigned to prevent autograd error when differentiating
        intrinsic[-1] = torch.atan2(euclid[-2] / cum_prod, euclid[-1] / cum_prod)
    return intrinsic


def to_extrinsic(intrinsic: torch.Tensor) -> torch.Tensor:
    n = intrinsic.shape[0]
    euclid = torch.zeros((n + 1,))

    if n == 1:  # implying extrinsic is at least dim 2
        euclid[0] = torch.cos(intrinsic[0])
        euclid[1] = torch.sin(intrinsic[0])
    else:  # implying extrinsic is at least dim 3
        euclid[0] = torch.cos(intrinsic[0])
        cum_prod = torch.sin(intrinsic[0])
        for i in range(1, n - 1):
            euclid[i] = torch.cos(intrinsic[i]) * cum_prod  # re-assigned
            cum_prod = cum_prod * torch.sin(intrinsic[i])
        euclid[-2] = torch.sin(intrinsic[-1]) * cum_prod
        euclid[-1] = torch.cos(intrinsic[-1]) * cum_prod

    return euclid


def _switch_antipodal_coords(coords: torch.ndarray, switch_coords: List[bool]) -> torch.Tensor:
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


def metric(intrinsic: torch.Tensor) -> torch.Tensor:
    coord_jacs = jacobian(to_extrinsic, intrinsic, create_graph=True)
    g = torch.tensordot(coord_jacs, coord_jacs, dims=([0], [0]))
    return g


def christoffels(intrinsic: torch.Tensor) -> torch.Tensor:
    g = metric(intrinsic)
    g_partials = jacobian(metric, intrinsic, create_graph=True)  # adds index at end due to partials

    # computes the connection coefficients of the Levi-Civita connection using the metric thereby describing the
    # curvature of the n-dimensional hypershere in the intrinsic coordinate system
    conn_coeffs = 0.5 * torch.tensordot(g.inverse(), g_partials + torch.transpose(g_partials, 1, 2) - torch.transpose(
        torch.transpose(g_partials, 1, 2), 0, 1), dims=([1], [0]))

    return conn_coeffs
