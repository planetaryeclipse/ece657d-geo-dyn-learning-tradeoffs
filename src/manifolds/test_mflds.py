import pytest

import itertools

import math

import torch
import torch.testing as tt

from pytest import approx

from typing import Tuple, Optional

import src.manifolds.sn_mfld as sn_mfld
from manifolds.coord_sys import ManifoldCoordSystem


def test_axis_permute():
    assert sn_mfld._axis_permute(0, 1) == (0,)

    n2_perms = list(itertools.permutations([0, 1]))
    assert sn_mfld._axis_permute(0, 2) == n2_perms[0]
    assert sn_mfld._axis_permute(1, 2) == n2_perms[1]

    n3_perms = list(itertools.permutations([0, 1, 2]))
    assert sn_mfld._axis_permute(0, 3) == n3_perms[0]
    assert sn_mfld._axis_permute(1, 3) == n3_perms[1]
    assert sn_mfld._axis_permute(2, 3) == n3_perms[2]
    assert sn_mfld._axis_permute(3, 3) == n3_perms[3]
    assert sn_mfld._axis_permute(4, 3) == n3_perms[4]
    assert sn_mfld._axis_permute(5, 3) == n3_perms[5]


def test_permute_idx_from_permutation():
    assert sn_mfld._permute_idx_from_permutation((0,), 1) == 0

    for i, permute in enumerate(itertools.permutations([0, 1])):
        assert sn_mfld._permute_idx_from_permutation(tuple(permute), 2) == i

    for i, permute in enumerate(itertools.permutations([0, 1, 2])):
        assert sn_mfld._permute_idx_from_permutation(tuple(permute), 3) == i


def test_non_singular_chart_ids():
    assert (sn_mfld._non_singular_chart_id(_unit(torch.tensor([1.0, 0.0]))) ==
            sn_mfld._permute_idx_from_permutation((1, 0), 2))
    assert (sn_mfld._non_singular_chart_id(_unit(torch.tensor([-1.0, 0.0]))) ==
            sn_mfld._permute_idx_from_permutation((0, 1), 2))

    assert (sn_mfld._non_singular_chart_id(_unit(torch.tensor([0.0, 1.0]))) ==
            sn_mfld._permute_idx_from_permutation((0, 1), 2))
    assert (sn_mfld._non_singular_chart_id(_unit(torch.tensor([0.0, -1.0]))) ==
            sn_mfld._permute_idx_from_permutation((1, 0), 2))

    # print("\nexperimentation...\n")
    # chart_idx = sn_mfld._non_singular_chart_id(_unit(torch.tensor([1.0, 0.0, 0.0])))
    # print(f"chart_idx: {chart_idx}")


def _unit(x, radius: float = 1.0):
    return radius * x / torch.linalg.norm(x)


# def test_switch_antipodal_coords():
#     tt.assert_close(sn_mfld._switch_antipodal_coords(torch.tensor([0.0]), [True]),
#                     torch.tensor([torch.pi]))
#     tt.assert_close(sn_mfld._switch_antipodal_coords(torch.tensor([torch.pi / 4]), [True]),
#                     torch.tensor([-3.0 / 4 * torch.pi]))
#     tt.assert_close(sn_mfld._switch_antipodal_coords(torch.tensor([-torch.pi / 4]), [True]),
#                     torch.tensor([+3.0 / 4 * torch.pi]))
#     tt.assert_close(sn_mfld._switch_antipodal_coords(torch.tensor([0.0, 0.0]), [False, True]),
#                     torch.tensor([0.0, torch.pi]))


@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_extrinsic_endomorphism_transform_at_singular_points(radius: float):
    # S1 embedding into R2
    x_set = [
        _unit(torch.tensor([1.0, 0.0]), radius),
        _unit(torch.tensor([0.0, 1.0]), radius),

        _unit(torch.tensor([-1.0, 0.0]), radius),
        _unit(torch.tensor([0.0, -1.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(2)):
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, chart_idx), chart_idx, radius), x)

    # S2 embedding into R3
    x_set = [
        _unit(torch.tensor([1.0, 0.0, 0.0]), radius),
        _unit(torch.tensor([-1.0, 0.0, 0.0]), radius),

        _unit(torch.tensor([0.0, 1.0, 0.0]), radius),
        _unit(torch.tensor([0.0, -1.0, 0.0]), radius),

        _unit(torch.tensor([0.0, 0.0, 1.0]), radius),
        _unit(torch.tensor([0.0, 0.0, -1.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(3)):
        intrinsic = sn_mfld.to_intrinsic(x, chart_idx)
        recon_extrinsic = sn_mfld.to_extrinsic(intrinsic, chart_idx, radius)

        tt.assert_close(recon_extrinsic, x)


@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_intrinsic_endomorphism_transform(radius: float):
    # S1 embedding into R2
    u_set = [
        torch.tensor([0.0]),
        torch.tensor([torch.pi])
    ]
    for u, chart_idx in itertools.product(u_set, range(2)):
        extrinsic = sn_mfld.to_extrinsic(u, chart_idx, radius)
        recon_intrinsic = sn_mfld.to_intrinsic(extrinsic, chart_idx)
        tt.assert_close(recon_intrinsic, u)

    # S2 embedding into R3
    u_set = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, torch.pi]),
        # torch.tensor([torch.pi, 0.0]), # multiple coords available -> reconstruction creates (pi, pi)
        # torch.tensor([torch.pi, torch.pi]), # multiple coords available -> reconstruction creates (pi, 0)
    ]
    for u, chart_idx in itertools.product(u_set, range(3)):
        extrinsic = sn_mfld.to_extrinsic(u, chart_idx, radius)
        recon_intrinsic = sn_mfld.to_intrinsic(extrinsic, chart_idx)

        # print(f"u: {u}, extrinsic: {extrinsic}, recon_intrinsic: {recon_intrinsic}")

        tt.assert_close(recon_intrinsic, u)


@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_extrinsic_endomorphism_transform(radius: float):
    # there is probably a better way to parameterize this test, but I find it helpful to visualize what tests are
    # actually being performed in converting the coordinates of the hypersphere

    # S1 embedding into R2
    x_set = [
        _unit(torch.tensor([3.0, 4.0]), radius),
        _unit(torch.tensor([-3.0, 4.0]), radius),
        _unit(torch.tensor([3.0, -4.0]), radius),
        _unit(torch.tensor([-3.0, -4.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(2)):
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, chart_idx), chart_idx, radius), x)

    # S2 embedding into R3
    x_set = [
        _unit(torch.tensor([3.0, 4.0, 5.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, 5.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(3)):
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, chart_idx), chart_idx, radius), x)

    # S3 embedding into R4
    x_set = [
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0]), radius),

        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(4)):
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, chart_idx), chart_idx, radius), x)

    # S4 embedding into R5
    x_set = [
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0, -7.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(5)):
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, chart_idx), chart_idx, radius), x)

    # S5 embedding into R6 (note that this increases the number of dimensions handled by the for loop internal to the
    # algorithm for switching to intrinsic and extrinsic coordinates so we do not need to test beyond this dimension)
    x_set = [
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, 4.0, -5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([3.0, -4.0, -5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, 4.0, -5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.tensor([-3.0, -4.0, -5.0, -6.0, -7.0, -8.0]), radius),
    ]
    for x, chart_idx in itertools.product(x_set, range(6)):
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, chart_idx), chart_idx, radius), x)


# @pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
# def test_intrinsic_ts_basis_in_extrinsic(radius):
#     # S1 embedding into R2
#     u_set = [
#         torch.tensor([0.0]),
#         torch.tensor([torch.pi])
#     ]
#     for u, chart_idx in itertools.product(u_set, range(2)):
#         basis = sn_mfld._intrinsic_ts_basis_in_extrinsic(u, chart_idx, radius)
#         assert torch.linalg.matrix_rank(basis) == 1
#
#         # extrinsic = sn_mfld.to_extrinsic(u, chart_idx, radius)
#         # recon_intrinsic = sn_mfld.to_intrinsic(extrinsic, chart_idx, radius)
#         # tt.assert_close(recon_intrinsic, u)
#
#     # S2 embedding into R3
#     u_set = [
#         torch.tensor([0.0, 0.0]),
#         torch.tensor([0.0, torch.pi]),
#         # torch.tensor([torch.pi, 0.0]), # multiple coords available -> reconstruction creates (pi, pi)
#         # torch.tensor([torch.pi, torch.pi]), # multiple coords available -> reconstruction creates (pi, 0)
#     ]
#     for u, chart_idx in itertools.product(u_set, range(3)):
#         basis = sn_mfld._intrinsic_ts_basis_in_extrinsic(u, chart_idx, radius)
#
#         print(f"u: {u}, chart_idx: {chart_idx}, basis {basis}")
#
#         assert torch.linalg.matrix_rank(basis) == 2
#
#         # extrinsic = sn_mfld.to_extrinsic(u, chart_idx, radius)
#         # recon_intrinsic = sn_mfld.to_intrinsic(extrinsic, chart_idx, radius)
#         #
#         # # print(f"u: {u}, extrinsic: {extrinsic}, recon_intrinsic: {recon_intrinsic}")
#         #
#         # tt.assert_close(recon_intrinsic, u)


# @pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
# def test_extrinsic_ts_isomorphism_at_singularity(radius):
#     # NOTE: we cannot test reliably from any intrinsic to extrinsic due to the singularity where there are multiple
#     # equally valid sets of angles when at the singular conditions
#
#     # S1 embedding into R2
#     x_ts_set = [
#         (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
#         (torch.tensor([1.0, 0.0]), torch.tensor([0.0, -1.0])),
#
#         (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])),
#         (torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])),
#
#         (torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 1.0])),
#         (torch.tensor([-1.0, 0.0]), torch.tensor([0.0, -1.0])),
#
#         (torch.tensor([0.0, -1.0]), torch.tensor([1.0, 0.0])),
#         (torch.tensor([0.0, -1.0]), torch.tensor([-1.0, 0.0])),
#     ]
#     for (x, x_ts), chart_idx in itertools.product(x_ts_set, range(2)):
#         intrinsic = sn_mfld.to_intrinsic(x, chart_idx)
#         intrinsic_ts = sn_mfld.to_intrinsic_ts(x, x_ts, chart_idx, radius)
#         recon_extrinsic_ts = sn_mfld.to_extrinsic_ts(intrinsic, intrinsic_ts, chart_idx, radius)
#
#         tt.assert_close(recon_extrinsic_ts, x_ts)
#
#     # S2 embedding into R3
#     x_ts_set = [
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])),
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0])),
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -1.0])),
#
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 1.0])),
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, -1.0])),
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 1.0])),
#         (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, -1.0])),
#     ]
#     for (x, x_ts), chart_idx in itertools.product(x_ts_set, range(3)):
#         intrinsic = sn_mfld.to_intrinsic(x, chart_idx)
#         intrinsic_ts = sn_mfld.to_intrinsic_ts(x, x_ts, chart_idx, radius)
#         recon_extrinsic_ts = sn_mfld.to_extrinsic_ts(intrinsic, intrinsic_ts, chart_idx, radius)
#
#         tt.assert_close(recon_extrinsic_ts, x_ts)

@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 2.0, 0.5], [0, 1]))
def test_s1_intrinsic_ts_isomorphism(radius, chart_idx: int):
    # S1 embedding into R2
    u_ts_set = [
        (torch.tensor([0.0]), torch.tensor([0.0])),
        (torch.tensor([0.0]), torch.tensor([1.0])),
        (torch.tensor([0.0]), torch.tensor([-1.0])),

        (torch.tensor([torch.pi / 2]), torch.tensor([0.0])),
        (torch.tensor([torch.pi / 2]), torch.tensor([1.0])),
        (torch.tensor([torch.pi / 2]), torch.tensor([-1.0])),

        (torch.tensor([torch.pi]), torch.tensor([0.0])),
        (torch.tensor([torch.pi]), torch.tensor([1.0])),
        (torch.tensor([torch.pi]), torch.tensor([-1.0])),

        (torch.tensor([3.0 / 2 * torch.pi]), torch.tensor([0.0])),
        (torch.tensor([3.0 / 2 * torch.pi]), torch.tensor([1.0])),
        (torch.tensor([3.0 / 2 * torch.pi]), torch.tensor([-1.0])),
    ]
    for (u, u_vec), chart_idx in itertools.product(u_ts_set, range(2)):
        try:
            extrinsic = sn_mfld.to_extrinsic(u, chart_idx, radius)
            extrinsic_vec = sn_mfld.to_extrinsic_ts(u, u_vec, chart_idx, radius)
            recon_intrinsic_vec = sn_mfld.to_intrinsic_ts(extrinsic, extrinsic_vec, chart_idx, radius)

            tt.assert_close(recon_intrinsic_vec, u_vec)
        except ValueError as e:
            print(
                f"\tSkipping as degenerate tangent basis in current chart: p_intrinsic={u}, chart_idx={chart_idx}")


@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 2.0, 0.5], [0]))  # list(range(math.factorial(3)))))
def test_s2_log_ts_isomorphism(radius, chart_idx: int):
    s2 = sn_mfld.HypersphereManifold(2, radius)
    chart = s2.charts[chart_idx]

    p_extrinsic = _unit(torch.tensor([1.0, 1.0, 1.0]), radius)
    q_extrinsic = _unit(torch.tensor([1.0, 1.0, -1.0]), radius)

    print(f"p_extrinsic: {p_extrinsic}, q_extrinsic: {q_extrinsic}")

    p_intrinsic = s2.to_intrinsic(chart, p_extrinsic)
    q_intrinsic = s2.to_intrinsic(chart, q_extrinsic)

    p_extrinsic_reconstructed = s2.to_extrinsic(chart, p_intrinsic)
    q_extrinsic_reconstructed = s2.to_extrinsic(chart, q_intrinsic)

    print(
        f"p_extrinsic_reconstructed: {p_extrinsic_reconstructed}, q_extrinsic_reconstructed: {q_extrinsic_reconstructed}")

    # problem occurs after here

    v_intrinsic = s2.log(chart, p_intrinsic, q_intrinsic)
    v_extrinsic = s2.to_extrinsic_ts(chart, p_intrinsic, v_intrinsic)

    print(f"v_extrinsic norm: {torch.norm(v_extrinsic)}")
    print(f"v_extrinsic: {v_extrinsic}")

    tt.assert_close(torch.norm(v_extrinsic), radius * torch.arccos(torch.tensor(1.0 / 3)))


@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 2.0, 0.5], list(range(math.factorial(3)))))
def test_s2_intrinsic_ts_isomorphism(radius, chart_idx: int):
    # S2 embedding into R3
    u_ts_set = [
        # first quadrant for first angle
        (torch.tensor([0.1, 0.0]), torch.tensor([0.0, 0.0])),
        (torch.tensor([0.1, 0.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.1, 0.0]), torch.tensor([0.0, -1.0])),
        (torch.tensor([0.1, 0.0]), torch.tensor([1.0, 1.0])),
        (torch.tensor([0.1, 0.0]), torch.tensor([1.0, -1.0])),
        (torch.tensor([0.1, 0.0]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([0.1, 0.0]), torch.tensor([-1.0, -1.0])),

        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([0.0, 0.0])),
        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([0.0, -1.0])),
        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([1.0, 1.0])),
        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([1.0, -1.0])),
        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([0.1, torch.pi / 2]), torch.tensor([-1.0, -1.0])),

        (torch.tensor([0.1, torch.pi]), torch.tensor([0.0, 0.0])),
        (torch.tensor([0.1, torch.pi]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.1, torch.pi]), torch.tensor([0.0, -1.0])),
        (torch.tensor([0.1, torch.pi]), torch.tensor([1.0, 1.0])),
        (torch.tensor([0.1, torch.pi]), torch.tensor([1.0, -1.0])),
        (torch.tensor([0.1, torch.pi]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([0.1, torch.pi]), torch.tensor([-1.0, -1.0])),

        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([0.0, 0.0])),
        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([0.0, -1.0])),
        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([1.0, 1.0])),
        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([1.0, -1.0])),
        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([0.1, 3.0 / 2 * torch.pi]), torch.tensor([-1.0, -1.0])),

        # second quadrant for second angle
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([0.0, 0.0])),
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([0.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([1.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, 0.0]), torch.tensor([-1.0, -1.0])),

        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([0.0, 0.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([0.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([0.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([1.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi / 2]), torch.tensor([-1.0, -1.0])),

        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([0.0, 0.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([0.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([0.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([1.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, torch.pi]), torch.tensor([-1.0, -1.0])),

        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([0.0, 0.0])),
        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([0.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([0.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([1.0, -1.0])),
        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([torch.pi - 0.1, 3.0 / 2 * torch.pi]), torch.tensor([-1.0, -1.0])),
    ]
    for (u, u_vec) in u_ts_set:
        try:
            extrinsic = sn_mfld.to_extrinsic(u, chart_idx, radius)
            extrinsic_vec = sn_mfld.to_extrinsic_ts(u, u_vec, chart_idx, radius)
            recon_intrinsic_vec = sn_mfld.to_intrinsic_ts(extrinsic, extrinsic_vec, chart_idx, radius)

            tt.assert_close(recon_intrinsic_vec, u_vec)
        except ValueError as e:
            print(
                f"\tSkipping as degenerate tangent basis in current chart: p_intrinsic={u}, chart_idx={chart_idx}")


@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_metric(radius):
    chart_idx = 0  # original formulation

    # S1 embedding into R2
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.tensor([3.0, 4.0]), radius), chart_idx)
    tt.assert_close(sn_mfld.metric(x_intrinsic, chart_idx, radius), radius ** 2 * torch.ones((1, 1)))

    # S2 embedding into R3
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.tensor([3.0, 4.0, 5.0]), radius), chart_idx)
    s2_metric = torch.tensor(
        [
            [radius ** 2, 0.0],
            [0.0, radius ** 2 * torch.sin(x_intrinsic[0]) ** 2]
        ])
    print(s2_metric)
    print(sn_mfld.metric(x_intrinsic, chart_idx, radius))
    tt.assert_close(sn_mfld.metric(x_intrinsic, chart_idx, radius), s2_metric)


@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_christoffels(radius):
    chart_idx = 0  # original formulation

    # S1 embedding into R2
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.tensor([3.0, 4.0]), radius), chart_idx)
    tt.assert_close(sn_mfld.christoffels(x_intrinsic, chart_idx, radius), torch.zeros((1, 1, 1)))

    # S2 embedding into R3
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.tensor([3.0, 4.0, 5.0]), radius), chart_idx)
    s2_christoffels = torch.zeros((2, 2, 2))
    s2_christoffels[0, 1, 1] = -torch.cos(x_intrinsic[0]) * torch.sin(x_intrinsic[0])
    s2_christoffels[1, 0, 1] = 1.0 / torch.tan(x_intrinsic[0])
    s2_christoffels[1, 1, 0] = s2_christoffels[1, 0, 1]
    tt.assert_close(sn_mfld.christoffels(x_intrinsic, chart_idx, radius), s2_christoffels)


@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 0.5, 2.0], [0, 1]))
def test_s1_distance(radius, chart_idx):
    s1 = sn_mfld.HypersphereManifold(1, radius)
    chart = s1.charts[chart_idx]

    # tests at itself
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([1.0, 0.0]))) == approx(0.0)

    # tests distances between singular points
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([-1.0, 0.0]))) == approx(radius * torch.pi)
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([-1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([1.0, 0.0]))) == approx(radius * torch.pi)

    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([0.0, 1.0])),
                       s1.to_intrinsic(chart, torch.tensor([0.0, -1.0]))) == approx(radius * torch.pi)
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([0.0, -1.0])),
                       s1.to_intrinsic(chart, torch.tensor([0.0, 1.0]))) == approx(radius * torch.pi)

    # tests halfway points
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([0.0, 1.0]))) == approx(radius * torch.pi / 2.0)
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([0.0, -1.0]))) == approx(radius * torch.pi / 2.0)
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([-1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([0.0, 1.0]))) == approx(radius * torch.pi / 2.0)
    assert s1.distance(chart,
                       s1.to_intrinsic(chart, torch.tensor([-1.0, 0.0])),
                       s1.to_intrinsic(chart, torch.tensor([0.0, -1.0]))) == approx(radius * torch.pi / 2.0)


@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 0.5, 2.0], [0]))
def test_s2_distance(radius, chart_idx):
    s2 = sn_mfld.HypersphereManifold(2, radius)
    chart = s2.charts[chart_idx]

    # tests at itself
    assert s2.distance(chart,
                       s2.to_intrinsic(chart, torch.tensor([1.0, 0.0, 0.0])),
                       s2.to_intrinsic(chart, torch.tensor([1.0, 0.0, 0.0]))) == 0.0

    # test antipodal point
    assert s2.distance(chart,
                       s2.to_intrinsic(chart, torch.tensor([1.0, 0.0, 0.0])),
                       s2.to_intrinsic(chart, torch.tensor([-1.0, 0.0, 0.0]))) == approx(radius * torch.pi)

    # tests the non-axial points
    assert s2.distance(chart,
                       s2.to_intrinsic(chart, _unit(torch.tensor([1.0, 1.0, 1.0]), radius)),
                       s2.to_intrinsic(chart, _unit(torch.tensor([1.0, 1.0, -1.0]), radius))) == approx(
        radius * torch.arccos(torch.tensor(1.0 / 3)))

    pass


@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 2.0, 0.5], [0, 1]))
def test_log_s1(radius, chart_idx):
    s1 = sn_mfld.HypersphereManifold(1, radius)
    chart = s1.charts[chart_idx]

    # for S1 the chart directions are reversed
    rev_factor = 1.0 if chart_idx == 0 else -1.0

    # NOTE: the distance being compared against is computed using a radius of 1.0 as this is a quantity measured in the
    # intrinsic coordinates, if the velocity is computed in extrinsic coordinates then it will be scaled by radius

    x_set = [
        # nonsingular points

        (s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.2]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)),
         torch.tensor([rev_factor * sn_mfld.distance(
             s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.2]), radius)),
             s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)), chart_idx, 1.0)])),
        (s1.to_intrinsic(chart, _unit(torch.tensor([1.0, -0.2]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)),
         torch.tensor([rev_factor * -sn_mfld.distance(
             s1.to_intrinsic(chart, _unit(torch.tensor([1.0, -0.2]), radius)),
             s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)), chart_idx, 1.0)])),
        # flipped direction
        (s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.2]), radius)),
         torch.tensor([rev_factor * -sn_mfld.distance(
             s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.2]), radius)),
             s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)), chart_idx, 1.0)])),
        (s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([1.0, -0.2]), radius)),
         torch.tensor([rev_factor * sn_mfld.distance(
             s1.to_intrinsic(chart, _unit(torch.tensor([1.0, -0.2]), radius)),
             s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)), chart_idx, 1.0)])),
        # between singular points

        (s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.0]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)),
         torch.tensor(
             [sn_mfld.distance(  # no rev factor as antipodal points will generate with ones tensor in intrinsic
                 s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.0]), radius)),
                 s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)), chart_idx, 1.0)])),
        # flipped direction
        (s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.0]), radius)),
         torch.tensor(
             [sn_mfld.distance(  # no rev factor as antipodal points will generate with ones tensor in intrinsic
                 s1.to_intrinsic(chart, _unit(torch.tensor([1.0, 0.0]), radius)),
                 s1.to_intrinsic(chart, _unit(torch.tensor([-1.0, 0.0]), radius)), chart_idx, 1.0)])),

        # across singular points

        (s1.to_intrinsic(chart, _unit(torch.tensor([0.2, 1.0]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([0.2, -1.0]), radius)),
         torch.tensor([rev_factor * -sn_mfld.distance(
             s1.to_intrinsic(chart, _unit(torch.tensor([0.2, 1.0]), radius)),
             s1.to_intrinsic(chart, _unit(torch.tensor([0.2, -1.0]), radius)), chart_idx, 1.0)])),
        # flipped
        (s1.to_intrinsic(chart, _unit(torch.tensor([0.2, -1.0]), radius)),
         s1.to_intrinsic(chart, _unit(torch.tensor([0.2, 1.0]), radius)),
         torch.tensor([rev_factor * sn_mfld.distance(
             s1.to_intrinsic(chart, _unit(torch.tensor([0.2, 1.0]), radius)),
             s1.to_intrinsic(chart, _unit(torch.tensor([0.2, -1.0]), radius)), chart_idx, 1.0)]))
    ]
    print()
    for i, (p, q, v) in enumerate(x_set):
        print(f"i: {i}")
        log = s1.log(chart, p, q)
        print(f"\tp: {p}, q: {q}, v: {v}, log: {log}")
        tt.assert_close(log, v)


def _log_s2_test_cfg(mfld: ManifoldCoordSystem, chart_idx: int, p_extrinsic: torch.Tensor, q_extrinsic: torch.Tensor,
                     v_dir_extrinsic: Optional[torch.Tensor] = None, v_dir_intrinsic: Optional[torch.Tensor] = None) -> \
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    chart = mfld.charts[chart_idx]
    p_intrinsic = mfld.to_intrinsic(chart, p_extrinsic)
    q_intrinsic = mfld.to_intrinsic(chart, q_extrinsic)

    # note that this method can fail if p_extrinsic is a singularity of the hypersphere in the specified chart so
    # we just return None for the intrinsic velocity and skip the test

    ts_basis_in_extrinsic = sn_mfld._intrinsic_ts_basis_in_extrinsic(p_intrinsic, chart_idx)
    try:
        sn_mfld._check_ts_basis(chart_idx, ts_basis_in_extrinsic)

        # some tests need to be provided the intrinsic value directly (to match the format used when generating the
        # value of the riemannian logarithm for antipodal points) so if extrinsic is not defined then use the intrinsic
        v_dir_intrinsic = mfld.to_intrinsic_ts(chart, p_extrinsic,
                                               v_dir_extrinsic) if v_dir_extrinsic is not None else v_dir_intrinsic
        v_dir_intrinsic_norm = torch.linalg.norm(v_dir_intrinsic)
        if v_dir_intrinsic_norm > 1E-6:  # basically so not dividing by a zero norm
            v_dir_intrinsic /= v_dir_intrinsic_norm
        dist = sn_mfld.distance(p_intrinsic, q_intrinsic, chart_idx, radius=1.0)
        v_dir_intrinsic *= dist
    except ValueError as e:
        v_dir_intrinsic = None
    return p_intrinsic, q_intrinsic, v_dir_intrinsic


@pytest.mark.parametrize('radius,chart_idx', itertools.product([1.0, 2.0, 0.5], list(range(math.factorial(3)))))
def test_log_s2(radius, chart_idx: int):
    s2 = sn_mfld.HypersphereManifold(2, radius)
    chart = s2.charts[chart_idx]

    x_set = [
        # check starting at (1,0,0)
        # checks comparisons with axes
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, 0.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 0.0]),
                         v_dir_intrinsic=torch.tensor([1.0, 1.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, 0.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, 0.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, -1.0])),

        # checks following around horizontal and vertical loops
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, 0.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, 0.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, 0.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, -1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, 0.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, 0.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, -1.0, 0.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, 0.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, 1.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, -1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, -1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, -1.0])),

        # checks following around diagonal loops
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, -1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, -1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, -1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, -1.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 1.0, -1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, -1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, 1.0])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, -1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, -1.0, -1.0])),

        # checks diagonal positions
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([0.0, 0.0, 0.0])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([2.0 / 3, 2.0 / 3, -4.0 / 3])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, -1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([2.0 / 3, -4.0 / 3, 2.0 / 3])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-1.0, 1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([-4.0 / 3, 2.0 / 3, 2.0 / 3])),

        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-1.0, -1.0, 1.0]),
                         v_dir_extrinsic=torch.tensor([-2.0 / 3, -2.0 / 3, 4.0 / 3])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-1.0, 1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([-2.0 / 3, 4.0 / 3, -2.0 / 3])),
        _log_s2_test_cfg(s2, chart_idx, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, -1.0, -1.0]),
                         v_dir_extrinsic=torch.tensor([4.0 / 3, -2.0 / 3, -2.0 / 3])),
    ]
    print()
    for i, (p_intrinsic, q_intrinsic, v_intrinsic) in enumerate(x_set):
        print(f"i: {i}")
        if v_intrinsic is None:
            print(
                f"\tSkipping as degenerate tangent basis in current chart: p_intrinsic={p_intrinsic}, chart_idx={chart_idx}")
        else:
            log = s2.log(chart, p_intrinsic, q_intrinsic)
            print(
                f"\tp_intrinsic: {p_intrinsic}, q_intrinsic: {q_intrinsic},\n\tv_intrinsic: {v_intrinsic}, log: {log}")
            tt.assert_close(log, v_intrinsic)

# def test_stuff():
#     x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.tensor([3.0, 4.0, 5.0, 6.0])))
#     metric = sn_mfld.metric(x_intrinsic)
#     christoffels = sn_mfld.christoffels(x_intrinsic)
#
#     print(f"metric: {metric}")
#     print(f"christoffels: {christoffels}")
