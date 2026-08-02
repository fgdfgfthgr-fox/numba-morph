import numpy as np
import pytest
from numba_morph import local_minima, local_maxima
from skimage import data, morphology
from utils import connectivity_to_footprint

class TestLMin2D:
    
    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    @pytest.mark.parametrize("fscale", [1.0, 10.0, 100.0, 1000.0])
    def test_lmin_2d_grey(self, connectivity, dtype, fscale):
        original = np.random.randint(0, 128, (32, 64), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((32, 64)).astype(dtype) * fscale
        footprint = connectivity_to_footprint(connectivity)
        result = local_minima(original, footprint=footprint)
        expected = morphology.local_minima(original, footprint=footprint, allow_borders=True).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    def test_lmin_2d_batched(self, connectivity, dtype):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((16, 32, 64)).astype(dtype)
        footprint = connectivity_to_footprint(connectivity)
        result = local_minima(original, footprint=footprint)
        expected = np.empty_like(original).astype(np.bool_)
        for i in range(expected.shape[0]):
            expected[i] = morphology.local_minima(original[i], footprint=footprint, allow_borders=True).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    def test_lmin_2d_coin(self, connectivity, dtype):
        original = data.coins()
        footprint = connectivity_to_footprint(connectivity)
        result = local_minima(original, footprint=footprint)
        expected = morphology.local_minima(original, footprint=footprint, allow_borders=True).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

class TestHMin3D:

    @pytest.mark.parametrize("connectivity", [6, 16, 26])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    def test_lmin_3d_grey(self, connectivity, dtype):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((16, 32, 64)).astype(dtype)
        footprint = connectivity_to_footprint(connectivity)
        result = local_minima(original, footprint=footprint)
        expected = morphology.local_minima(original, footprint=footprint, allow_borders=True).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 16, 26])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    def test_lmin_3d_batched(self, connectivity, dtype):
        original = np.random.randint(0, 128, (8, 16, 32, 64), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((8, 16, 32, 64)).astype(dtype)
        footprint = connectivity_to_footprint(connectivity)
        result = local_minima(original, footprint=footprint)
        expected = np.empty_like(original).astype(np.bool_)
        for i in range(expected.shape[0]):
            expected[i] = morphology.local_minima(original[i], footprint=footprint, allow_borders=True).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)


class TestLMax2D:
    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    def test_lmax_2d_grey(self, connectivity, dtype):
        original = np.random.randint(0, 128, (32, 64), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((32, 64)).astype(dtype)
        footprint = connectivity_to_footprint(connectivity)
        result = local_maxima(original, footprint=footprint)
        expected = morphology.local_maxima(original, footprint=footprint, allow_borders=True).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)