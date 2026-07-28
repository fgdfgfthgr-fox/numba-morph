import numpy as np
import pytest
import scipy.ndimage as ndimage
from numba_morph import opening, closing
from skimage import data
from utils import connectivity_to_footprint

class TestOpening2D:

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_opening_2d_grey(self, connectivity, mode):
        original = np.random.randint(0, 128, (32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_opening(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_opening_2d_binary(self, connectivity):
        original = np.random.randint(0, 2, (256, 512), dtype=np.bool_)
        mask = np.random.randint(0, 2, (256, 512), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mask=mask, mode='constant')
        expected = ndimage.binary_opening(original, structure=footprint, mask=mask, border_value=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_opening_2d_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_opening(original, footprint=footprint, axes=(1, 2), mode=mode)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_opening_2d_coin(self, connectivity, mode):
        original = data.coins()
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_opening(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

class TestOpening3D:

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_opening_3d_grey(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_opening(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_opening_3d_binary(self, connectivity):
        original = np.random.randint(0, 2, (32, 64, 128), dtype=np.bool_)
        mask = np.random.randint(0, 2, (32, 64, 128), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mask=mask, mode='constant')
        expected = ndimage.binary_opening(original, structure=footprint, mask=mask, border_value=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_opening_3d_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (8, 16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = opening(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_opening(original, footprint=footprint, axes=(1, 2, 3), mode=mode)
        np.testing.assert_array_equal(result, expected)


class TestClosing2D:

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_closing_2d(self, connectivity, mode):
        original = np.random.randint(0, 128, (32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = closing(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_closing(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_closing_2d_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = closing(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_closing(original, footprint=footprint, axes=(1, 2), mode=mode)
        np.testing.assert_array_equal(result, expected)

class TestClosing3D:

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_closing_3d(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = closing(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_closing(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_opening_3d_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (8, 16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = closing(original, footprint=footprint, mode=mode)
        expected = ndimage.grey_closing(original, footprint=footprint, axes=(1, 2, 3), mode=mode)
        np.testing.assert_array_equal(result, expected)