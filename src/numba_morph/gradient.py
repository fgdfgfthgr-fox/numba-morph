import numpy as np
from .dilation import dilation
from .erosion import erosion

def morphological_gradient(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0):
    """
        Multidimensional morphological gradient.
        The morphological gradient is calculated as the difference between a dilation and an erosion of the input
        with a given structuring element.

        Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
        The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
        Whether to choose 2D or 3D operation is determined by the size or footprint.

        Parameters
        ----------
        input : array_like
            Input.
        size : tuple of ints, optional
            Shape of a flat and full structuring element used for the morphology operations.
            Optional if `footprint` or `structure` is provided.
            A larger `size` yields a more blurred gradient.
        footprint : array of ints, optional
            The neighborhood expressed as an n-D array of 1’s and 0’s.
            i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
        structure : array of ints, optional
            Structuring element used for the gradient.
            Note structure is not yet supported!
        output : array, optional
            An array used for storing the output of the gradient may be provided.
        mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
            Determines how the array borders are handled. Default is 'reflect'.
        cval : scalar, optional
            Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.

        Returns
        -------
        result : ndarray
            Morphological gradient of `input`.
        """
    if input.dtype == np.bool:
        minus = np.bitwise_xor
    else:
        minus = np.subtract
    tmp = dilation(input, size=size, footprint=footprint, structure=structure, mode=mode, cval=cval)
    if isinstance(output, np.ndarray):
        erosion(input, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval)
        return minus(tmp, output, output)
    else:
        return minus(tmp, erosion(input, size=size, footprint=footprint, structure=structure, mode=mode, cval=cval))