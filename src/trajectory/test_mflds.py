import unittest

import torch
import torch.testing as tt

import trajectory.sn_mfld as sn_mfld


def _unit(x):
    return x / torch.linalg.norm(x)


class TestSnManifold(unittest.TestCase):
    def test_switch_antipodal_coords(self):
        tt.assert_close(sn_mfld._switch_antipodal_coords(torch.tensor([0.0]), [True]),
                        torch.Tensor([torch.pi]))
        tt.assert_close(sn_mfld._switch_antipodal_coords(torch.Tensor([torch.pi / 4]), [True]),
                        torch.Tensor([-3.0 / 4 * torch.pi]))
        tt.assert_close(sn_mfld._switch_antipodal_coords(torch.Tensor([-torch.pi / 4]), [True]),
                        torch.Tensor([+3.0 / 4 * torch.pi]))
        tt.assert_close(sn_mfld._switch_antipodal_coords(torch.Tensor([0.0, 0.0]), [False, True]),
                        torch.Tensor([0.0, torch.pi]))

    def test_intrinsic_extrinsic_transform(self):
        # S1 embedding into R2
        x_set = [
            _unit(torch.Tensor([3.0, 4.0])),
            _unit(torch.Tensor([-3.0, 4.0])),
            _unit(torch.Tensor([3.0, -4.0])),
            _unit(torch.Tensor([-3.0, -4.0])),
        ]
        for x in x_set:
            tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

        # S2 embedding into R3
        x_set = [
            _unit(torch.Tensor([3.0, 4.0, 5.0])),
            _unit(torch.Tensor([3.0, 4.0, -5.0])),
            _unit(torch.Tensor([3.0, -4.0, 5.0])),
            _unit(torch.Tensor([3.0, -4.0, -5.0])),

            _unit(torch.Tensor([-3.0, 4.0, 5.0])),
            _unit(torch.Tensor([-3.0, 4.0, -5.0])),
            _unit(torch.Tensor([-3.0, -4.0, 5.0])),
            _unit(torch.Tensor([-3.0, -4.0, -5.0])),
        ]
        for x in x_set:
            tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

        # S3 embedding into R4
        x_set = [
            _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0])),
            _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0])),
            _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0])),
            _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0])),

            _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0])),
            _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0])),
            _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0])),
            _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0])),

            _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0])),
            _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0])),
            _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0])),
            _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0])),

            _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0])),
            _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0])),
            _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0])),
            _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0])),
        ]
        for x in x_set:
            tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

        # S4 embedding into R5
        x_set = [
            _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, 7.0])),
            _unit(torch.Tensor([3.0, 4.0, 5.0, 6.0, -7.0])),
            _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, 7.0])),
            _unit(torch.Tensor([3.0, 4.0, 5.0, -6.0, -7.0])),

            _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, 7.0])),
            _unit(torch.Tensor([3.0, 4.0, -5.0, 6.0, -7.0])),
            _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, 7.0])),
            _unit(torch.Tensor([3.0, 4.0, -5.0, -6.0, -7.0])),

            _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, 7.0])),
            _unit(torch.Tensor([3.0, -4.0, 5.0, 6.0, -7.0])),
            _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, 7.0])),
            _unit(torch.Tensor([3.0, -4.0, 5.0, -6.0, -7.0])),

            _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, 7.0])),
            _unit(torch.Tensor([3.0, -4.0, -5.0, 6.0, -7.0])),
            _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, 7.0])),
            _unit(torch.Tensor([3.0, -4.0, -5.0, -6.0, -7.0])),

            _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, 7.0])),
            _unit(torch.Tensor([-3.0, 4.0, 5.0, 6.0, -7.0])),
            _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, 7.0])),
            _unit(torch.Tensor([-3.0, 4.0, 5.0, -6.0, -7.0])),

            _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, 7.0])),
            _unit(torch.Tensor([-3.0, 4.0, -5.0, 6.0, -7.0])),
            _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, 7.0])),
            _unit(torch.Tensor([-3.0, 4.0, -5.0, -6.0, -7.0])),

            _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, 7.0])),
            _unit(torch.Tensor([-3.0, -4.0, 5.0, 6.0, -7.0])),
            _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, 7.0])),
            _unit(torch.Tensor([-3.0, -4.0, 5.0, -6.0, -7.0])),

            _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, 7.0])),
            _unit(torch.Tensor([-3.0, -4.0, -5.0, 6.0, -7.0])),
            _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, 7.0])),
            _unit(torch.Tensor([-3.0, -4.0, -5.0, -6.0, -7.0])),
        ]
        for x in x_set:
            tt.assert_close(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

    def test_metric(self):
        # S1 embedding into R2
        x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0])))
        tt.assert_close(sn_mfld.metric(x_intrinsic), torch.ones((1, 1)))

        # S2 embedding into R3
        x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0, 5.0])))
        s2_metric = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, torch.sin(x_intrinsic[0])**2]
            ])
        tt.assert_close(sn_mfld.metric(x_intrinsic), s2_metric)

    def test_christoffels(self):
        # S1 embedding into R2
        x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0])))
        tt.assert_close(sn_mfld.christoffels(x_intrinsic), torch.zeros((1, 1, 1)))

        # S2 embedding into R3
        x_intrinsic = sn_mfld.to_intrinsic(_unit(torch.Tensor([3.0, 4.0, 5.0])))
        s2_christoffels = torch.zeros((2,2,2))
        s2_christoffels[0, 1, 1] = -torch.cos(x_intrinsic[0]) * torch.sin(x_intrinsic[0])
        s2_christoffels[1, 0, 1] = 1.0 / torch.tan(x_intrinsic[0])
        s2_christoffels[1, 1, 0] = s2_christoffels[1, 0, 1]
        tt.assert_close(sn_mfld.christoffels(x_intrinsic), s2_christoffels)


if __name__ == '__main__':
    unittest.main()
