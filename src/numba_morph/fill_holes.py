# Not implemented because https://github.com/seung-lab/fill_voids exist.


'''import numpy as np
from .dilation import dilation

def fill_holes(input, footprint=None, mode='reflect', cval=0):
    """
    Fill the holes in binary objects. Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the size or footprint.

    Parameters
    ----------
    input : ndarray
        N-D binary array with holes to be filled
    footprint : ndarray
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. default is 'reflect'.
    cval : int
        The value when mode is equal to 'constant'. Default is 0.

    Returns
    -------
    result : ndarray
        Transformation of the initial image input where holes have been filled.
    """
    mask = np.logical_not(input)
    tmp = np.zeros(mask.shape, bool)
    output = dilation(tmp, footprint=footprint, mask=mask, mode=mode, cval=cval)
    np.logical_not(output, output)
    return output'''