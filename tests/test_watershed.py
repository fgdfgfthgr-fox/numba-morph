import numpy as np
import pytest
import skimage.segmentation as segmentation
from numba_morph import watershed
from utils import connectivity_to_footprint

class TestWatershed:

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int16])
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_watershed_2d(self, connectivity, dtype):
        image = np.random.randint(0, 255, size=(100, 100), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((100, 100)).astype(dtype)
        markers = np.zeros((100, 100), dtype=np.int32)
        markers[10, 10] = 1
        markers[90, 90] = 2
        footprint = connectivity_to_footprint(connectivity)
        result = watershed(image, markers=markers, footprint=footprint)
        expected = segmentation.watershed(image, markers=markers, connectivity=footprint)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int16])
    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_watershed_2d_batched(self, connectivity, dtype):
        image = np.random.randint(0, 255, size=(2, 100, 100), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((2, 100, 100)).astype(dtype)
        markers = np.zeros((2, 100, 100), dtype=np.int32)
        markers[0, 10, 10] = 1
        markers[0, 90, 90] = 2
        markers[1, 20, 20] = 1
        markers[1, 80, 80] = 2
        footprint = connectivity_to_footprint(connectivity)
        result = watershed(image, markers=markers, footprint=footprint)
        expected = np.zeros_like(result)
        for i in range(result.shape[0]):
            expected[i] = segmentation.watershed(image[i], markers=markers[i], connectivity=footprint)
        np.testing.assert_array_equal(result, expected)


    @pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int16])
    @pytest.mark.parametrize("connectivity", [6, 18, 26])
    def test_watershed_3d(self, connectivity, dtype):
        image = np.random.randint(0, 255, size=(100, 100, 100), dtype=dtype) if not np.issubdtype(dtype, np.floating) else np.random.random((100, 100, 100)).astype(dtype)
        markers = np.zeros((100, 100, 100), dtype=np.int32)
        markers[10, 10, 10] = 1
        markers[90, 90, 90] = 2
        markers[40, 40, 40] = 3
        footprint = connectivity_to_footprint(connectivity)
        result = watershed(image, markers=markers, footprint=footprint)
        expected = segmentation.watershed(image, markers=markers, connectivity=footprint)
        np.testing.assert_array_equal(result, expected)