import numpy as np
import pytest
import scipy.ndimage as ndimage
from numba_morph import morphological_gradient
from skimage import data
from utils import connectivity_to_footprint

class TestGradient2D:

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_gradient_2d_grey(self, connectivity, mode):
        original = np.random.randint(0, 128, (32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint, mode=mode)
        expected = ndimage.morphological_gradient(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

    '''@pytest.mark.parametrize("connectivity", [4, 8])
    def test_gradient_2d_binary(self, connectivity):
        original = np.random.randint(0, 2, (256, 512), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint)
        expected = ndimage.morphological_gradient(original, structure=footprint)
        np.testing.assert_array_equal(result, expected)'''

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_gradient_2d_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint, mode=mode)
        expected = ndimage.morphological_gradient(original, footprint=footprint, axes=(1, 2), mode=mode)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_gradient_2d_coin(self, connectivity, mode):
        original = data.coins()
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint, mode=mode)
        expected = ndimage.morphological_gradient(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

class TestGradient3D:

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_gradient_3d_grey(self, connectivity, mode):
        original = np.random.randint(0, 128, (16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint, mode=mode)
        expected = ndimage.morphological_gradient(original, footprint=footprint, mode=mode)
        np.testing.assert_array_equal(result, expected)

    '''@pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_gradient_3d_binary(self, connectivity):
        original = np.random.randint(0, 2, (32, 64, 128), dtype=np.bool_)
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint)
        expected = ndimage.morphological_gradient(original, structure=footprint)
        np.testing.assert_array_equal(result, expected)'''

    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_gradient_3d_batched(self, connectivity, mode):
        original = np.random.randint(0, 128, (8, 16, 32, 64), dtype=np.uint8)
        footprint = connectivity_to_footprint(connectivity)
        result = morphological_gradient(original, footprint=footprint, mode=mode)
        expected = ndimage.morphological_gradient(original, footprint=footprint, axes=(1, 2, 3), mode=mode)
        np.testing.assert_array_equal(result, expected)

