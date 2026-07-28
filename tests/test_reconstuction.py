import numpy as np
import pytest
import skimage.morphology as morph
from numba_morph import reconstruction
from skimage import data
from utils import connectivity_to_footprint

# 2D cases
class TestReconstruction2D:
    @pytest.mark.parametrize("speed", [True, False])
    @pytest.mark.parametrize("mode", ["erosion", "dilation"])
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_reconstruction_2d(self, connectivity, mode, speed):
        original = np.random.randint(low=0, high=255, size=(64, 64), dtype=np.uint8)
        seed = np.copy(original)
        if mode == "erosion":
            seed[1:-1, 1:-1] = original.max()
        elif mode == "dilation":
            seed[1:-1, 1:-1] = original.min()
        footprint = connectivity_to_footprint(connectivity)
        expected = morph.reconstruction(seed, original, mode, footprint).astype(np.uint8)
        result = reconstruction(original, seed, method=mode, footprint=footprint, speed=speed)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("speed", [True, False])
    @pytest.mark.parametrize("mode", ["erosion", "dilation"])
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_reconstruction_2d_batched(self, connectivity, mode, speed):
        original = np.random.randint(low=0, high=255, size=(8, 64, 64), dtype=np.uint8)
        seed = np.copy(original)
        if mode == "erosion":
            seed[1:-1, 1:-1] = original.max()
        elif mode == "dilation":
            seed[1:-1, 1:-1] = original.min()
        footprint = connectivity_to_footprint(connectivity)
        expected = np.empty_like(seed)
        for i in range(expected.shape[0]):
            expected[i] = morph.reconstruction(seed[i], original[i], mode, footprint).astype(np.uint8)
        result = reconstruction(original, seed, method=mode, footprint=footprint, speed=speed)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("speed", [True, False])
    @pytest.mark.parametrize("mode", ["erosion", "dilation"])
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_coin(self, connectivity, mode, speed):
        """One of the examples in skimage's 'Filtering regional maxima'"""
        original = data.coins()
        seed = np.copy(original)
        if mode == "erosion":
            seed[1:-1, 1:-1] = original.max()
        elif mode == "dilation":
            seed[1:-1, 1:-1] = original.min()
        footprint = connectivity_to_footprint(connectivity)
        expected = morph.reconstruction(seed, original, mode, footprint).astype(np.uint8)
        result = reconstruction(original, seed, method=mode, footprint=footprint, speed=speed)
        np.testing.assert_array_equal(result, expected)


# 3D cases
class TestReconstruction3D:
    @pytest.mark.parametrize("speed", [True, False])
    @pytest.mark.parametrize("mode", ["erosion", "dilation"])
    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_reconstruction_3d(self, connectivity, mode, speed):
        original = np.random.randint(low=0, high=255, size=(32, 32, 32), dtype=np.uint8)
        seed = np.copy(original)
        if mode == "erosion":
            seed[1:-1, 1:-1] = original.max()
        elif mode == "dilation":
            seed[1:-1, 1:-1] = original.min()
        footprint = connectivity_to_footprint(connectivity)
        expected = morph.reconstruction(seed, original, mode, footprint).astype(np.uint8)
        result = reconstruction(original, seed, method=mode, footprint=footprint, speed=speed)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("speed", [True, False])
    @pytest.mark.parametrize("mode", ["erosion", "dilation"])
    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_reconstruction_3d_batched(self, connectivity, mode, speed):
        original = np.random.randint(low=0, high=255, size=(8, 32, 32, 32), dtype=np.uint8)
        seed = np.copy(original)
        if mode == "erosion":
            seed[1:-1, 1:-1] = original.max()
        elif mode == "dilation":
            seed[1:-1, 1:-1] = original.min()
        footprint = connectivity_to_footprint(connectivity)
        expected = np.empty_like(seed)
        for i in range(expected.shape[0]):
            expected[i] = morph.reconstruction(seed[i], original[i], mode, footprint).astype(np.uint8)
        result = reconstruction(original, seed, method=mode, footprint=footprint, speed=speed)
        np.testing.assert_array_equal(result, expected)

