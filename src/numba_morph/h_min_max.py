import warnings
import numpy as np
from .reconstruction import reconstruction

def compare_greater_equal_safe(a, b, h, block_size=4096):
    """returns True where a >= b + h, avoiding overflow/underflow"""
    if not np.issubdtype(a.dtype, np.unsignedinteger):
        return a - b >= h

    result = np.empty(a.shape, dtype=bool)

    # Flatten to 1D for easy indexing (view, no copy)
    flat_a = a.ravel()
    flat_b = b.ravel()
    flat_res = result.ravel()
    n = flat_a.size

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_a = flat_a[start:end]
        block_b = flat_b[start:end]

        # Local computation for this block
        added = block_b + h          # block‑sized temporary
        overflow = added < block_b   # block‑sized boolean
        block_res = (block_a >= added) & ~overflow

        flat_res[start:end] = block_res

    return result

def h_maxima(input, h, footprint=None, mode="reflect", cval=0.0):
    """
        Determine all maxima of the image with height >= h.

        The local maxima are defined as connected sets of pixels with equal gray level strictly greater than the gray
        level of all pixels in direct neighborhood of the set.

        A local maximum M of height h is a local maximum for which there is at least one path joining M with an equal or
        higher local maximum on which the minimal value is f(M) - h (i.e. the values along the path are not decreasing
        by more than h with respect to the maximum’s value) and no path to an equal or higher local maximum for which
        the minimal value is greater.

        The global maxima of the image are also found by this function.

        Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
        The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
        Whether to choose 2D or 3D operation is determined by the size or footprint.

        Parameters
        ----------
        input : array_like
            Input.
        h : int
            The minimal height of all extracted maxima.
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
    if h > np.ptp(input):
        return np.zeros(input.shape, dtype=np.bool_)
    if np.issubdtype(type(h), np.floating) and np.issubdtype(input.dtype, np.integer):
        if (h % 1) != 0:
            warnings.warn(
                'possible precision loss converting image to '
                'floating point. To silence this warning, '
                'ensure image and h have same data type.',
                stacklevel=2,
            )
            input = input.astype(float)
        else:
            h = input.dtype.type(h)

    if h == 0:
        raise ValueError("h = 0 is ambiguous, use skimage.morphology.local_maxima() " "instead?")

    if np.issubdtype(input.dtype, np.floating):
        # To get the purpose of resolution, check the original code in skimage.morphology.h_maxima
        resolution = 2 * np.finfo(input.dtype).resolution * np.abs(input)
        h = h + resolution

    rec_img = reconstruction(input, dynamic=-h, method='dilation', footprint=footprint, edge_mode=mode, cval=cval)
    # returns True where a >= b + h, avoiding overflow/underflow
    return compare_greater_equal_safe(input, rec_img, h)


def h_minima(input, h, footprint=None, mode="reflect", cval=0.0):
    """
        Determine all minima of the image with height >= h.

        The local minima are defined as connected sets of pixels with equal gray level strictly smaller than the gray
        level of all pixels in direct neighborhood of the set.

        A local minimum M of depth h is a local minimum for which there is at least one path joining M with an equal or
        lower local minimum on which the maximal value is f(M) + h (i.e. the values along the path are not increasing
        by more than h with respect to the minimum's value) and no path to an equal or lower local minimum for which
        the maximal value is smaller.

        The global minima of the image are also found by this function.

        Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
        The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
        Whether to choose 2D or 3D operation is determined by the size or footprint.

        Parameters
        ----------
        input : array_like
            Input.
        h : int
            The minimal depth of all extracted minima.
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
    if h > np.ptp(input):
        return np.zeros(input.shape, dtype=np.bool_)
    if np.issubdtype(type(h), np.floating) and np.issubdtype(input.dtype, np.integer):
        if (h % 1) != 0:
            warnings.warn(
                'possible precision loss converting image to '
                'floating point. To silence this warning, '
                'ensure image and h have same data type.',
                stacklevel=2,
            )
            input = input.astype(float)
        else:
            h = input.dtype.type(h)

    if h == 0:
        raise ValueError("h = 0 is ambiguous, use skimage.morphology.local_minima() " "instead?")

    if np.issubdtype(input.dtype, np.floating):
        # To get the purpose of resolution, check the original code in skimage.morphology.h_maxima
        resolution = np.finfo(input.dtype).eps * max(input.min(), input.max(), key=abs)
        h = h + resolution

    rec_img = reconstruction(input, dynamic=h, method='erosion', footprint=footprint, edge_mode=mode, cval=cval)
    return compare_greater_equal_safe(rec_img, input, h)