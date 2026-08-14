import numpy as np
import pytest
import scipy.ndimage as ndimage
from numba_morph import distance_transform_cdt

class TestDistanceTransform2D:

    @pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
    def test_chamfer_distance_transform_2d(self, dtype):
        original = np.random.randint(0, 2, (32, 64), dtype=np.uint8)
        result = distance_transform_cdt(original, weights=(1,2), dtype=dtype)
        expected = ndimage.distance_transform_cdt(original, 'taxicab')
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
    def test_chamfer_distance_transform_2d_batched(self, dtype):
        original = np.random.randint(0, 2, (16, 32, 64), dtype=np.uint8)
        result = distance_transform_cdt(original, weights=(1,2), dtype=dtype)
        expected = np.empty_like(original)
        for i in range(expected.shape[0]):
            expected[i] = ndimage.distance_transform_cdt(original[i], 'taxicab')
        np.testing.assert_array_equal(result, expected)

class TestDistanceTransform3D:

    @pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
    def test_chamfer_distance_transform_3d(self, dtype):
        original = np.random.randint(0, 2, (16, 32, 64), dtype=np.uint8)
        result = distance_transform_cdt(original, weights=(1,2,3), dtype=dtype)
        expected = ndimage.distance_transform_cdt(original, 'taxicab')
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
    def test_chamfer_distance_transform_3d_batched(self, dtype):
        original = np.random.randint(0, 2, (8, 16, 32, 64), dtype=np.uint8)
        result = distance_transform_cdt(original, weights=(1,2,3), dtype=dtype)
        expected = np.empty_like(original)
        for i in range(expected.shape[0]):
            expected[i] = ndimage.distance_transform_cdt(original[i], 'taxicab')
        np.testing.assert_array_equal(result, expected)