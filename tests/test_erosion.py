import numpy as np
import pytest
import scipy.ndimage as ndimage
from numba_morph import erosion
from utils import connectivity_to_footprint

class TestErosion2D:

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_erosion_2d_grey(self, connectivity, mode):
        original = np.random.randint(0, 128, (32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_erosion(original, footprint=footprint, mode=mode, cval=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_erosion_2d_binary(self, connectivity):
        original = np.random.randint(0, 2, (64, 128), dtype=np.bool_)
        mask = np.random.randint(0, 2, (64, 128), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mask=mask, mode='constant')
        expected = ndimage.binary_erosion(original, structure=footprint, mask=mask, border_value=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_erosion_2d_grey_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_erosion(original, footprint=footprint, axes=(1, 2), mode=mode, cval=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_erosion_2d_binary_batched(self, connectivity):
        original = np.random.randint(0, 2, (8, 64, 128), dtype=np.bool_)
        mask = np.random.randint(0, 2, (8, 64, 128), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mask=mask, mode='constant')
        expected = ndimage.binary_erosion(original, structure=footprint, mask=mask, border_value=0, axes=(1, 2))
        np.testing.assert_array_equal(result, expected)

class TestErosion3D:

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_erosion_3d_grey(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_erosion(original, footprint=footprint, mode=mode, cval=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_erosion_3d_binary(self, connectivity):
        original = np.random.randint(0, 2, (32, 64, 128), dtype=np.bool_)
        mask = np.random.randint(0, 2, (32, 64, 128), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mask=mask, mode='constant')
        expected = ndimage.binary_erosion(original, structure=footprint, mask=mask, border_value=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_erosion_3d_grey_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (8, 16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_erosion(original, footprint=footprint, axes=(1, 2, 3), mode=mode, cval=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_erosion_3d_binary_batched(self, connectivity):
        original = np.random.randint(0, 2, (8, 32, 64, 128), dtype=np.bool_)
        mask = np.random.randint(0, 2, (8, 32, 64, 128), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = erosion(original, footprint=footprint, mask=mask, mode='constant')
        expected = ndimage.binary_erosion(original, structure=footprint, mask=mask, border_value=0, axes=(1, 2, 3))
        np.testing.assert_array_equal(result, expected)