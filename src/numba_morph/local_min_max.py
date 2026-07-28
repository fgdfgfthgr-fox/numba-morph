import numpy as np
import warnings
from .h_min_max import h_minima, h_maxima

def local_maxima(input, footprint=None, allow_borders=True):
    """
        Find local maxima of n-dimensional array.

        The local maxima are defined as connected sets of pixels with equal gray level (plateaus) strictly greater than
        the gray level of all pixels in direct neighborhood of the set.

        Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
        The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
        Whether to choose 2D or 3D operation is determined by the footprint.

        Parameters
        ----------
        input : array_like
            Input.
        footprint : array of ints, optional
            The neighborhood expressed as an n-D array of 1’s and 0’s.
            i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
        allow_borders : bool, optional
            If true, plateaus that touch the image border are valid maxima.

        Returns
        -------
        result : ndarray
            A binary image in bool, where pixels belonging to the determined maxima take value 1, the others take value 0.
    """
    if np.issubdtype(input.dtype, np.floating):
        raise NotImplementedError('numba-morph does not yet support using floating-point types in local maxima!')

    h = np.finfo(input.dtype).resolution if np.issubdtype(input.dtype, np.floating) else 1
    if allow_borders:
        return h_maxima(input, h, footprint=footprint)
    else:
        info = np.finfo(input.dtype) if np.issubdtype(input.dtype, np.floating) else np.iinfo(input.dtype)
        max_val = info.max
        warnings.warn('Due to how local maxima in numba-morph works, setting allow_borders to False may not always '
                      'remove plateaus from borders! It is unlikely as it require the pixel value in the plateau to '
                      'be the maximum value of the dtype, but it can happen in extreme cases.')
        return h_maxima(input, h, footprint=footprint, mode='constant', cval=max_val)


def local_minima(input, footprint=None, allow_borders=True):
    """
        Find local minima of n-dimensional array.

        The local minima are defined as connected sets of pixels with equal gray level (plateaus) strictly smaller than
        the gray level of all pixels in direct neighborhood of the set.

        Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
        The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
        Whether to choose 2D or 3D operation is determined by the footprint.

        Parameters
        ----------
        input : array_like
            Input.
        footprint : array of ints, optional
            The neighborhood expressed as an n-D array of 1’s and 0’s.
            i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
        allow_borders : bool, optional
            If true, plateaus that touch the image border are valid minima.

        Returns
        -------
        result : ndarray
            A binary image in bool, where pixels belonging to the determined minima take value 1, the others take value 0.
    """
    if np.issubdtype(input.dtype, np.floating):
        raise NotImplementedError('numba-morph does not yet support using floating-point types in local maxima!')
    h = np.finfo(input.dtype).eps if np.issubdtype(input.dtype, np.floating) else 1
    if allow_borders:
        return h_minima(input, h, footprint=footprint)
    else:
        info = np.finfo(input.dtype) if np.issubdtype(input.dtype, np.floating) else np.iinfo(input.dtype)
        min_val = info.min
        warnings.warn('Due to how local minima in numba-morph works, setting allow_borders to False may not always '
                      'remove plateaus from borders! It is unlikely as it require the pixel value in the plateau to '
                      'be the minimal value of the dtype, but it can happen in extreme cases.')
        return h_minima(input, h, footprint=footprint, mode='constant', cval=min_val)