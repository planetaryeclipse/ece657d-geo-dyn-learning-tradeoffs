import pytest

import torch
import torch.testing as tt

import sn_mfld as sn_mfld

def _unit(x, radius: float = 1.0):
    return radius * x / torch.linalg.norm(x)

def test_switch_antipodal_coords():
    tt.assert_close(sn_mfld._switch_antipodal_coords(torch.tensor([0.0]), [True]),
                    torch.Tensor([torch.pi]))
    tt.assert_close(sn_mfld._switch_antipodal_coords(torch.Tensor([torch.pi / 4]), [True]),
                    torch.Tensor([-3.0 / 4 * torch.pi]))
    tt.assert_close(sn_mfld._switch_antipodal_coords(torch.Tensor([-torch.pi / 4]), [True]),
                    torch.Tensor([+3.0 / 4 * torch.pi]))
    tt.assert_close(sn_mfld._switch_antipodal_coords(torch.Tensor([0.0, 0.0]), [False, True]),
                    torch.Tensor([0.0, torch.pi]))


@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_intrinsic_extrinsic_transform(radius: float):
    # there is probably a better way to parameterize this test, but I find it helpful to visualize what tests are
    # actually being performed in converting the coordinates of the hypersphere

    # S1 embedding into R2
    x_set = [
        _unit(torch.Tensor([3.0, 4.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0]), radius),
        _unit(torch.Tensor([3.0, -4.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0]), radius),
    ]
    for x in x_set:
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, radius), radius), x)

    # S2 embedding into R3
    x_set = [
        _unit(torch.Tensor([3.0, 4.0, 5.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, 5.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0]), radius),
    ]
    for x in x_set:
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, radius), radius), x)

    # S3 embedding into R4
    x_set = [
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0]), radius),
    ]
    for x in x_set:
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, radius), radius), x)

    # S4 embedding into R5
    x_set = [
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, -7.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, -7.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, 7.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, -7.0]), radius),
    ]
    for x in x_set:
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, radius), radius), x)

    # S5 embedding into R6 (note that this increases the number of dimensions handled by the for loop internal to the
    # algorithm for switching to intrinsic and extrinsic coordinates so we do not need to test beyond this dimension)
    x_set = [
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, -7.0, -8.0]), radius),

        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, 7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, 7.0, -8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, -7.0, 8.0]), radius),
        _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, -7.0, -8.0]), radius),
    ]
    for x in x_set:
        tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x, radius), radius), x)

@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_metric(radius):
    # S1 embedding into R2
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0]), radius), radius)
    tt.assert_close(sn_mfld.metric(x_intrinsic, radius), radius ** 2 * torch.ones((1, 1)))

    # S2 embedding into R3
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0, 5.0]), radius), radius)
    s2_metric = torch.tensor(
        [
            [radius**2, 0.0],
            [0.0, radius**2 * torch.sin(x_intrinsic[0])**2]
        ])
    print(s2_metric)
    print(sn_mfld.metric(x_intrinsic, radius))
    tt.assert_close(sn_mfld.metric(x_intrinsic, radius), s2_metric)

@pytest.mark.parametrize('radius', [1.0, 2.0, 0.5])
def test_christoffels(radius):
    # S1 embedding into R2
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0]), radius), radius)
    tt.assert_close(sn_mfld.christoffels(x_intrinsic, radius), torch.zeros((1, 1, 1)))

    # S2 embedding into R3
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0, 5.0]), radius), radius)
    s2_christoffels = torch.zeros((2,2,2))
    s2_christoffels[0, 1, 1] = -torch.cos(x_intrinsic[0]) * torch.sin(x_intrinsic[0])
    s2_christoffels[1, 0, 1] = 1.0 / torch.tan(x_intrinsic[0])
    s2_christoffels[1, 1, 0] = s2_christoffels[1, 0, 1]
    tt.assert_close(sn_mfld.christoffels(x_intrinsic, radius), s2_christoffels)

def test_stuff():
    x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0, 5.0, 6.0])))
    metric = sn_mfld.metric(x_intrinsic)
    christoffels = sn_mfld.christoffels(x_intrinsic)

    print(f"metric: {metric}")
    print(f"christoffels: {christoffels}")