import unittest

import numpy as np
import numpy.testing as npt

import trajectory.sn_mfld as sn_mfld


def _unit(x):
    return x / np.linalg.norm(x)


class TestSnManifold(unittest.TestCase):
    def test_switch_antipodal_coords(self):
        npt.assert_allclose(sn_mfld._switch_antipodal_coords(np.array([0.0]), [True]),
                            np.array([np.pi]))
        npt.assert_allclose(sn_mfld._switch_antipodal_coords(np.array([np.pi / 4]), [True]),
                            np.array([-3.0 / 4 * np.pi]))
        npt.assert_allclose(sn_mfld._switch_antipodal_coords(np.array([-np.pi / 4]), [True]),
                            np.array([+3.0 / 4 * np.pi]))
        npt.assert_allclose(sn_mfld._switch_antipodal_coords(np.array([0.0, 0.0]), [False, True]),
                            np.array([0.0, np.pi]))

    def test_intrinsic_extrinsic_transform(self):
        # S1 embedding into R2
        x_set = [
            _unit(np.array([3.0, 4.0])),
            _unit(np.array([-3.0, 4.0])),
            _unit(np.array([3.0, -4.0])),
            _unit(np.array([-3.0, -4.0])),
        ]
        for x in x_set:
            npt.assert_allclose(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

        # S2 embedding into R3
        x_set = [
            _unit(np.array([3.0, 4.0, 5.0])),
            _unit(np.array([3.0, 4.0, -5.0])),
            _unit(np.array([3.0, -4.0, 5.0])),
            _unit(np.array([3.0, -4.0, -5.0])),

            _unit(np.array([-3.0, 4.0, 5.0])),
            _unit(np.array([-3.0, 4.0, -5.0])),
            _unit(np.array([-3.0, -4.0, 5.0])),
            _unit(np.array([-3.0, -4.0, -5.0])),
        ]
        for x in x_set:
            npt.assert_allclose(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

        # S3 embedding into R4
        x_set = [
            _unit(np.array([3.0, 4.0, 5.0, 6.0])),
            _unit(np.array([3.0, 4.0, 5.0, -6.0])),
            _unit(np.array([3.0, 4.0, -5.0, 6.0])),
            _unit(np.array([3.0, 4.0, -5.0, -6.0])),

            _unit(np.array([3.0, -4.0, 5.0, 6.0])),
            _unit(np.array([3.0, -4.0, 5.0, -6.0])),
            _unit(np.array([3.0, -4.0, -5.0, 6.0])),
            _unit(np.array([3.0, -4.0, -5.0, -6.0])),

            _unit(np.array([-3.0, 4.0, 5.0, 6.0])),
            _unit(np.array([-3.0, 4.0, 5.0, -6.0])),
            _unit(np.array([-3.0, 4.0, -5.0, 6.0])),
            _unit(np.array([-3.0, 4.0, -5.0, -6.0])),

            _unit(np.array([-3.0, -4.0, 5.0, 6.0])),
            _unit(np.array([-3.0, -4.0, 5.0, -6.0])),
            _unit(np.array([-3.0, -4.0, -5.0, 6.0])),
            _unit(np.array([-3.0, -4.0, -5.0, -6.0])),
        ]
        for x in x_set:
            npt.assert_allclose(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)

        # S4 embedding into R5
        x_set = [
            _unit(np.array([3.0, 4.0, 5.0, 6.0, 7.0])),
            _unit(np.array([3.0, 4.0, 5.0, 6.0, -7.0])),
            _unit(np.array([3.0, 4.0, 5.0, -6.0, 7.0])),
            _unit(np.array([3.0, 4.0, 5.0, -6.0, -7.0])),

            _unit(np.array([3.0, 4.0, -5.0, 6.0, 7.0])),
            _unit(np.array([3.0, 4.0, -5.0, 6.0, -7.0])),
            _unit(np.array([3.0, 4.0, -5.0, -6.0, 7.0])),
            _unit(np.array([3.0, 4.0, -5.0, -6.0, -7.0])),

            _unit(np.array([3.0, -4.0, 5.0, 6.0, 7.0])),
            _unit(np.array([3.0, -4.0, 5.0, 6.0, -7.0])),
            _unit(np.array([3.0, -4.0, 5.0, -6.0, 7.0])),
            _unit(np.array([3.0, -4.0, 5.0, -6.0, -7.0])),

            _unit(np.array([3.0, -4.0, -5.0, 6.0, 7.0])),
            _unit(np.array([3.0, -4.0, -5.0, 6.0, -7.0])),
            _unit(np.array([3.0, -4.0, -5.0, -6.0, 7.0])),
            _unit(np.array([3.0, -4.0, -5.0, -6.0, -7.0])),

            _unit(np.array([-3.0, 4.0, 5.0, 6.0, 7.0])),
            _unit(np.array([-3.0, 4.0, 5.0, 6.0, -7.0])),
            _unit(np.array([-3.0, 4.0, 5.0, -6.0, 7.0])),
            _unit(np.array([-3.0, 4.0, 5.0, -6.0, -7.0])),

            _unit(np.array([-3.0, 4.0, -5.0, 6.0, 7.0])),
            _unit(np.array([-3.0, 4.0, -5.0, 6.0, -7.0])),
            _unit(np.array([-3.0, 4.0, -5.0, -6.0, 7.0])),
            _unit(np.array([-3.0, 4.0, -5.0, -6.0, -7.0])),

            _unit(np.array([-3.0, -4.0, 5.0, 6.0, 7.0])),
            _unit(np.array([-3.0, -4.0, 5.0, 6.0, -7.0])),
            _unit(np.array([-3.0, -4.0, 5.0, -6.0, 7.0])),
            _unit(np.array([-3.0, -4.0, 5.0, -6.0, -7.0])),

            _unit(np.array([-3.0, -4.0, -5.0, 6.0, 7.0])),
            _unit(np.array([-3.0, -4.0, -5.0, 6.0, -7.0])),
            _unit(np.array([-3.0, -4.0, -5.0, -6.0, 7.0])),
            _unit(np.array([-3.0, -4.0, -5.0, -6.0, -7.0])),
        ]
        for x in x_set:
            npt.assert_allclose(sn_mfld.to_extrinsic(sn_mfld.to_intrinsic(x)), x)


if __name__ == '__main__':
    unittest.main()
