import numpy as np
from .open_close import closing, opening

def black_tophat(input, size=None, footprint=None, structure=None, output=None,
                 mode="reflect", cval=0.0):
    """
    Multidimensional black tophat filter.
    Defined as its morphological closing minus the original image.
    This operation returns the dark spots of the image that are smaller than the footprint.
    Note that dark spots in the original image are bright spots after the black top hat.

    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the size or footprint.

    Parameters
    ----------
    input : array_like
        Input.
    size : tuple of ints, optional
        Shape of a flat and full structuring element used for the filter.
        Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    structure : array of ints, optional
        Structuring element used for the filter.
        Note structure is not yet supported!
    output : array, optional
        An array used for storing the output of the filter may be provided.
    mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.

    Returns
    -------
    result : ndarray
        Result of the filter of `input` with `structure`.
    """
    tmp = closing(input, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval)
    if input.dtype == np.bool_ and tmp.dtype == np.bool_:
        np.bitwise_xor(tmp, input, out=tmp)
    else:
        np.subtract(tmp, input, out=tmp)
    return tmp

def white_tophat(input, size=None, footprint=None, structure=None, output=None,
                 mode="reflect", cval=0.0):
    """
    Multidimensional white tophat filter.
    Defined as the image minus its morphological opening.
    This operation returns the bright spots of the image that are smaller than the footprint.

    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the size or footprint.

    Parameters
    ----------
    input : array_like
        Input.
    size : tuple of ints, optional
        Shape of a flat and full structuring element used for the filter.
        Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    structure : array of ints, optional
        Structuring element used for the filter.
        Note structure is not yet supported!
    output : array, optional
        An array used for storing the output of the filter may be provided.
    mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.

    Returns
    -------
    result : ndarray
        Result of the filter of `input` with `structure`.
    """
    tmp = opening(input, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval)
    if input.dtype == np.bool_ and tmp.dtype == np.bool_:
        np.bitwise_xor(input, tmp, out=tmp)
    else:
        np.subtract(input, tmp, out=tmp)
    return tmp