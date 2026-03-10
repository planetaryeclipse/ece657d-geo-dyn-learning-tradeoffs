# Sn manifold (n-dimensional hypersphere smoothly embedded in Rn+1)
from math import prod

import numpy as np
import itertools

from typing import List


def to_intrinsic(euclid: np.ndarray) -> np.ndarray:
    euclid_n = euclid.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1

    intrinsic = np.zeros((n,))

    if n == 1:
        intrinsic[0] = np.atan2(euclid[1], euclid[0])
    else:
        intrinsic[0] = np.acos(euclid[0])
        cum_prod = np.sin(intrinsic[0])

        for i in range(1, n - 1):
            intrinsic[i] = np.acos(euclid[i] / cum_prod)
            cum_prod *= np.sin(intrinsic[i])
        intrinsic[-1] = np.atan2(euclid[-2] / cum_prod, euclid[-1]/cum_prod)
    return intrinsic


def to_extrinsic(intrinsic: np.ndarray) -> np.ndarray:
    n = intrinsic.shape[0]
    euclid = np.zeros((n + 1,))

    if n == 1:  # implying extrinsic is at least dim 2
        euclid[0] = np.cos(intrinsic[0])
        euclid[1] = np.sin(intrinsic[0])
    else:  # implying extrinsic is at least dim 3
        euclid[0] = np.cos(intrinsic[0])
        cum_prod = np.sin(intrinsic[0])
        for i in range(1, n - 1):
            euclid[i] = np.cos(intrinsic[i]) * cum_prod
            cum_prod *= np.sin(intrinsic[i])
        euclid[-2] = np.sin(intrinsic[-1]) * cum_prod
        euclid[-1] = np.cos(intrinsic[-1]) * cum_prod

    return euclid


def _switch_antipodal_coords(coords: np.ndarray, switch_coords: List[bool]) -> np.ndarray:
    continuous_coords = (coords + 2 * np.pi) - np.array([np.pi if switch else 0 for switch in switch_coords])
    switched_coords = np.array([coord if coord <= np.pi else -(2 * np.pi - coord) for coord in continuous_coords])

    return switched_coords


def to_other_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    n = intrinsic.shape[0]
    total_charts = 2 ** intrinsic.shape[0]  # antipodal chart for each coordinate

    intrinsic_charts = np.zeros((total_charts, n))
    for i, antipodal in enumerate(itertools.product([False, True], repeat=n)):
        intrinsic_charts[i, :] = _switch_antipodal_coords(intrinsic, antipodal)

    return intrinsic_charts
