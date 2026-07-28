from .erosion import erosion
from .dilation import dilation


def opening(input, size=None, footprint=None, structure=None, iterations=1, mask=None, output=None, mode='reflect', cval=0):
    """
    Calculate an opening operation for ndarray.
    Which consists in the succession of a grayscale erosion, and a grayscale dilation.
    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the size or footprint.

    Parameters
    ----------
    input : ndarray
        Array over which the opening is to be computed.
    size : tuple of int
        Shape of a flat and full structuring element used for the opening.
        Optional if footprint or structure is provided.
    footprint : ndarray
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    structure : ndarray
        Structuring element used for the opening.
        The structure array applies a subtractive offset for each pixel in the neighborhood.
        Note structure is not yet supported!
    iterations : int, optional
        The erosion step of the opening, then the dilation step are each repeated iterations times (one, by default).
        If iterations is less than 1, each operation is repeated until the result does not change anymore.
    mask : ndarray, optional
        If a mask is given, only those elements with a True value at the corresponding mask element are modified.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed. By default, a copy of the input array is created.
    mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. default is 'reflect'.
    cval : int
        The value when mode is equal to 'constant'. Default is 0.

    Returns
    -------
    result : ndarray
        Opening of input.
    """
    tmp = erosion(input, size, footprint, structure, iterations, mask, None, mode, cval)
    return dilation(tmp, size, footprint, structure, iterations, mask, output, mode, cval)


def closing(input, size=None, footprint=None, structure=None, iterations=1, mask=None, output=None, mode='reflect', cval=0):
    """
    Calculate an opening operation for ndarray.
    Which consists in the succession of a grayscale erosion, and a grayscale dilation.
    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the size or footprint.

    Parameters
    ----------
    input : ndarray
        Array over which the opening is to be computed.
    size : tuple of int
        Shape of a flat and full structuring element used for the opening.
        Optional if footprint or structure is provided.
    footprint : ndarray
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    structure : ndarray
        Structuring element used for the opening.
        The structure array applies a subtractive offset for each pixel in the neighborhood.
        Note structure is not yet supported!
    iterations : int, optional
        The dilation step of the closing, then the erosion step are each repeated iterations times (one, by default).
        If iterations is less than 1, each operation is repeated until the result does not change anymore.
    mask : ndarray, optional
        If a mask is given, only those elements with a True value at the corresponding mask element are modified.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed. By default, a new array is created.
    mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. default is 'reflect'.
    cval : int
        The value when mode is equal to 'constant'. Default is 0.

    Returns
    -------
    result : ndarray
        Opening of input.
    """
    tmp = dilation(input, size, footprint, structure, iterations, mask, None, mode, cval)
    return erosion(tmp, size, footprint, structure, iterations, mask, output, mode, cval)
