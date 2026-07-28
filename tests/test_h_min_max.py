import numpy as np
import pytest
from numba_morph import h_minima, h_maxima
from skimage import data, morphology
from utils import connectivity_to_footprint

class TestHMin2D:
    
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_hmin_2d_grey(self, connectivity):
        original = np.random.randint(0, 128, (32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = h_minima(original, 2, footprint=footprint, mode='constant', cval=127)
        expected = morphology.h_minima(original, 2, footprint=footprint).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_hmin_2d_batched(self, connectivity):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = h_minima(original, 2, footprint=footprint, mode='constant', cval=127)
        expected = np.empty_like(original).astype(np.bool_)
        for i in range(expected.shape[0]):
            expected[i] = morphology.h_minima(original[i], 2, footprint=footprint).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_hmin_2d_coin(self, connectivity):
        original = data.coins()
        footprint = connectivity_to_footprint(connectivity)
        data_max = original.max()
        result = h_minima(original, 5, footprint=footprint, mode='constant', cval=data_max)
        expected = morphology.h_minima(original, 5, footprint=footprint).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

class TestHMin3D:

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_hmin_3d_grey(self, connectivity):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = h_minima(original, 2, footprint=footprint, mode='constant', cval=127)
        expected = morphology.h_minima(original, 2, footprint=footprint).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_hmin_3d_batched(self, connectivity):
        original = np.random.randint(0, 128, (8, 16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = h_minima(original, 2, footprint=footprint, mode='constant', cval=127)
        expected = np.empty_like(original).astype(np.bool_)
        for i in range(expected.shape[0]):
            expected[i] = morphology.h_minima(original[i], 2, footprint=footprint).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)


class TestHMax2D:
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_hmax_2d_grey(self, connectivity):
        original = np.random.randint(0, 128, (32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = h_maxima(original, 2, footprint=footprint, mode='constant', cval=0)
        expected = morphology.h_maxima(original, 2, footprint=footprint).astype(np.bool_)
        np.testing.assert_array_equal(result, expected)