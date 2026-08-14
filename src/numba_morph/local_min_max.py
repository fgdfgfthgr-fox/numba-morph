import numpy as np
import warnings
from .utils import safe_add
from .reconstruction import reconstruction


def local_maxima(input, footprint=None, mode="reflect", cval=0.0):
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
        mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
            Determines how the array borders are handled. Default is 'reflect'.
        cval : scalar, optional
            Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.

        Returns
        -------
        result : ndarray
            A binary image in bool, where pixels belonging to the determined maxima take value 1, the others take value 0.
    """
    if np.issubdtype(input.dtype, np.floating):
        marker = np.nextafter(input, -np.inf)
    else:
        marker = safe_add(input, -1)
    reconstruction(input, marker, inplace=True, method='dilation', footprint=footprint, edge_mode=mode, cval=cval)
    return marker < input


def local_minima(input, footprint=None, mode="reflect", cval=0.0):
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
        mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
            Determines how the array borders are handled. Default is 'reflect'.
        cval : scalar, optional
            Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.

        Returns
        -------
        result : ndarray
            A binary image in bool, where pixels belonging to the determined minima take value 1, the others take value 0.
    """
    if np.issubdtype(input.dtype, np.floating):
        marker = np.nextafter(input, np.inf)
    else:
        marker = safe_add(input, 1)
    reconstruction(input, marker, inplace=True, method='erosion', footprint=footprint, edge_mode=mode, cval=cval)
    return marker > input