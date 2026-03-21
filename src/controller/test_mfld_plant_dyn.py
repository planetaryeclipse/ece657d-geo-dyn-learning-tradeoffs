import torch
import torch.testing as tt

import pytest
import itertools

import src.manifolds.sn_mfld as sn_mfld

from src.manifolds.test_mflds import _unit
from src.controller.mfld_plant_dyn import ManualManifoldPlantDynamics

from pytest import approx


@pytest.mark.parametrize("radius, p_extrinsic, q_extrinsic",
                         itertools.product([1.0, 2.0, 0.5],
                                           [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
                                            torch.tensor([-1.0, 0.0]), torch.tensor([0.0, -1.0])],
                                           [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])], ))
def test_s1_dynamics(radius, p_extrinsic, q_extrinsic):
    s1 = sn_mfld.HypersphereManifold(1, radius)

    p_extrinsic, q_extrinsic = _unit(p_extrinsic, radius), _unit(q_extrinsic, radius)
    chart = s1.nonsingular_chart_id(p_extrinsic)

    p_intrinsic, q_intrinsic = s1.to_intrinsic(chart, p_extrinsic), s1.to_intrinsic(chart, q_extrinsic)

    v_intrinsic = s1.log(chart, p_intrinsic, q_intrinsic)

    s1_dynamics = ManualManifoldPlantDynamics(s1,
                                              (chart, p_intrinsic.detach().numpy(), v_intrinsic.detach().numpy()),
                                              1)

    result = s1_dynamics.run_for(0.01, 1.0)
    tt.assert_close(torch.tensor(result.pos_extrinsic), q_extrinsic)


s2_test_points = [
    # quadrants
    torch.tensor([1.0, 1.0, 1.0]),
    torch.tensor([1.0, 1.0, -1.0]),
    torch.tensor([1.0, -1.0, 1.0]),
    torch.tensor([1.0, -1.0, -1.0]),

    torch.tensor([-1.0, 1.0, 1.0]),
    torch.tensor([-1.0, 1.0, -1.0]),
    torch.tensor([-1.0, -1.0, 1.0]),
    torch.tensor([-1.0, -1.0, -1.0]),

    # axes
    torch.tensor([1.0, 0.0, 0.0]),
    torch.tensor([0.0, 1.0, 0.0]),
    torch.tensor([0.0, 0.0, 1.0]),

    torch.tensor([-1.0, 0.0, 0.0]),
    torch.tensor([0.0, -1.0, 0.0]),
    torch.tensor([0.0, 0.0, -1.0]),

    # loop positions
    torch.tensor([1.0, 1.0, 0.0]),
    torch.tensor([1.0, -1.0, 0.0]),
    torch.tensor([-1.0, -1.0, 0.0]),
    torch.tensor([-1.0, 1.0, 0.0]),

    torch.tensor([1.0, 0.0, 1.0]),
    torch.tensor([1.0, 0.0, -1.0]),
    torch.tensor([-1.0, 0.0, -1.0]),
    torch.tensor([-1.0, 0.0, 1.0]),

    torch.tensor([0.0, 1.0, 1.0]),
    torch.tensor([0.0, 1.0, -1.0]),
    torch.tensor([0.0, -1.0, -1.0]),
    torch.tensor([0.0, -1.0, 1.0]),
]

s2_supp_test_points = [
    torch.tensor([-0.6124061, -0.55891754, -0.55907956]),
    torch.tensor([1.0, 1.0, 0.0])
]


@pytest.mark.parametrize("radius, p_extrinsic, q_extrinsic",
                         itertools.product([1.0],
                                           s2_test_points,
                                           s2_test_points))
def test_s2_dynamics(radius, p_extrinsic, q_extrinsic):
    s2 = sn_mfld.HypersphereManifold(2, radius)

    p_extrinsic, q_extrinsic = _unit(p_extrinsic, radius), _unit(q_extrinsic, radius)
    chart = s2.nonsingular_chart_id(p_extrinsic)

    p_intrinsic, q_intrinsic = s2.to_intrinsic(chart, _unit(p_extrinsic, radius)), s2.to_intrinsic(chart,
                                                                                                   _unit(q_extrinsic))

    print(f"p_extrinsic: {p_extrinsic}, q_extrinsic: {q_extrinsic}")
    print(f"p_intrinsic: {p_intrinsic}, q_intrinsic: {q_intrinsic}")

    v_intrinsic = s2.log(chart, p_intrinsic, q_intrinsic)
    print(f"v_intrinsic: {v_intrinsic}")
    v_extrinsic = s2.to_extrinsic_ts(chart, p_intrinsic, v_intrinsic)
    print(f"v_extrinsic: {v_extrinsic}")

    s2_dynamics = ManualManifoldPlantDynamics(s2,
                                              (chart, p_intrinsic.detach().numpy(), v_intrinsic.detach().numpy()),
                                              2)

    result = s2_dynamics.run_for(0.01, 1.0)

    print(f"result: {result}")

    # NOTE: there are so many floating point operations that long term geodesics are hard to compute (theoretically a
    # better architecture could be used but that's not needed for this project)
    tt.assert_close(torch.tensor(result.pos_extrinsic), q_extrinsic, rtol=1E-4, atol=1E-5)


s3_test_points = [
    # torch.tensor([1.0, 0.0, 0.0, 0.0]),
    # torch.tensor([0.0, 1.0, 0.0, 0.0]),
    # torch.tensor([0.0, 0.0, 1.0, 0.0]),
    # torch.tensor([0.0, 0.0, 0.0, 1.0]),
    #
    # torch.tensor([-1.0, 0.0, 0.0, 0.0]),
    # torch.tensor([0.0, -1.0, 0.0, 0.0]),
    # torch.tensor([0.0, 0.0, -1.0, 0.0]),
    # torch.tensor([0.0, 0.0, 0.0, -1.0]),

    torch.tensor([1.0, 1.0, 1.0, 1.0]),
    torch.tensor([1.0, 1.0, 1.0, -1.0]),
    torch.tensor([1.0, 1.0, -1.0, 1.0]),
    torch.tensor([1.0, 1.0, 1.0, -1.0]),

    # torch.tensor([1.0, -1.0, 1.0, 1.0]),
    # torch.tensor([1.0, -1.0, 1.0, -1.0]),
    # torch.tensor([1.0, -1.0, -1.0, 1.0]),
    # torch.tensor([1.0, -1.0, 1.0, -1.0]),

    torch.tensor([1.0, 1.0, 1.0, 0.0]),
    torch.tensor([1.0, 1.0, 0.0, 1.0]),
    torch.tensor([1.0, 0.0, 1.0, 1.0]),
    torch.tensor([0.0, 1.0, 1.0, 1.0]),
]


@pytest.mark.parametrize("radius, p_extrinsic, q_extrinsic",
                         itertools.product([1.0, 2.0, 0.5], s3_test_points, s3_test_points))
def test_s3_dynamics(radius, p_extrinsic, q_extrinsic):
    s3 = sn_mfld.HypersphereManifold(3, radius)

    p_extrinsic, q_extrinsic = _unit(p_extrinsic, radius), _unit(q_extrinsic, radius)
    chart = s3.nonsingular_chart_id(p_extrinsic)

    p_intrinsic, q_intrinsic = s3.to_intrinsic(chart, p_extrinsic), s3.to_intrinsic(chart, q_extrinsic)

    print(f"p_extrinsic: {p_extrinsic}, q_extrinsic: {q_extrinsic}")
    print(f"p_intrinsic: {p_intrinsic}, q_intrinsic: {q_intrinsic}")

    v_intrinsic = s3.log(chart, p_intrinsic, q_intrinsic)

    print(f"v_intrinsic: {v_intrinsic}")

    s3_dynamics = ManualManifoldPlantDynamics(s3,
                                              (chart, p_intrinsic.detach().numpy(), v_intrinsic.detach().numpy()),
                                              3)

    result = s3_dynamics.run_for(0.01, 1.0)

    print(f"result: {result}")

    print(f"result pos extrinsic: {result.pos_extrinsic.dtype}")
    print(f"q_extrinsic: {q_extrinsic.dtype}")

    # NOTE: there are so many floating point operations that long term geodesics are hard to compute (theoretically a
    # better architecture could be used but that's not needed for this project)
    tt.assert_close(torch.tensor(result.pos_extrinsic), q_extrinsic, rtol=1E-4, atol=1E-5)


@pytest.mark.parametrize("radius, p_extrinsic, q_extrinsic",
                         itertools.product([1.0, 2.0, 0.5],
                                           [torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
                                            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                                            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                                            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                                            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                                            torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0]),
                                            torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0]),
                                            torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0]),
                                            torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0]),
                                            torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0]),
                                            ],
                                           [torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                                            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])]))
def test_s4_dynamics(radius, p_extrinsic, q_extrinsic):
    s4 = sn_mfld.HypersphereManifold(4, radius)

    p_extrinsic, q_extrinsic = _unit(p_extrinsic, radius), _unit(q_extrinsic, radius)
    chart = s4.nonsingular_chart_id(p_extrinsic)

    p_intrinsic, q_intrinsic = s4.to_intrinsic(chart, p_extrinsic), s4.to_intrinsic(chart, q_extrinsic)

    print(f"p_extrinsic: {p_extrinsic}, q_extrinsic: {q_extrinsic}")
    print(f"p_intrinsic: {p_intrinsic}, q_intrinsic: {q_intrinsic}")

    v_intrinsic = s4.log(chart, p_intrinsic, q_intrinsic)

    print(f"v_intrinsic: {v_intrinsic}")

    s3_dynamics = ManualManifoldPlantDynamics(s4,
                                              (chart, p_intrinsic.detach().numpy(), v_intrinsic.detach().numpy()),
                                              4)

    result = s3_dynamics.run_for(0.01, 1.0)

    print(f"result: {result}")

    print(f"result pos extrinsic: {result.pos_extrinsic.dtype}")
    print(f"q_extrinsic: {q_extrinsic.dtype}")

    # NOTE: there are so many floating point operations that long term geodesics are hard to compute (theoretically a
    # better architecture could be used but that's not needed for this project)
    tt.assert_close(torch.tensor(result.pos_extrinsic), q_extrinsic, rtol=1E-4, atol=1E-5)


def test_stuff():
    s4 = sn_mfld.HypersphereManifold(4, 1.0)

    # torch.tensor([0.5, torch.pi / 2, torch.pi / 2, torch.pi / 2, torch.pi / 2])

    basis = sn_mfld._intrinsic_ts_basis_in_extrinsic(
        torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0]), 0, 1.0)
    print(f"basis:\n{basis}")

    close_basis = sn_mfld._intrinsic_ts_basis_in_extrinsic(
        torch.tensor([1.0, 0.001, 1.0, 1.0, 1.0]), 0, 1.0
    )

    print(f"rank: {torch.linalg.matrix_rank(basis)}")
    print(f"basis_sqr: {torch.diag(torch.tensordot(basis, basis, ([0], [0])))}")

    pass
