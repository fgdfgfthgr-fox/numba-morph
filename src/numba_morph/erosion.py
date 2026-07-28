import numpy as np
import scipy.ndimage as ndimage
from ._scan import _get_offsets, _scan_filter


def erosion(input, size=None, footprint=None, structure=None, iterations=1, mask=None, output=None,
            mode='reflect', cval=0, dilation=False):
    """
    Calculate an erosion operation for ndarray.
    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the size or footprint.

    Parameters
    ----------
    input : ndarray
        Array over which the grayscale erosion is to be computed.
    size : tuple of int
        Shape of a flat and full structuring element used for the grayscale erosion.
        Optional if footprint or structure is provided.
    footprint : ndarray
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    structure : ndarray
        Structuring element used for the grayscale erosion.
        The structure array applies a subtractive offset for each pixel in the neighborhood.
        Note structure is not yet supported!
    iterations : int, optional
        The erosion is repeated iterations times (one, by default).
        If iterations is less than 1, the erosion is repeated until the result does not change anymore.
    mask : ndarray, optional
        If a mask is given, only those elements with a True value at the corresponding mask element are modified.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed. By default, a copy of the input array is created.
    mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. default is 'reflect'.
    cval : int
        The value when mode is equal to 'constant'. Default is 0.
    dilation : bool
        If True, dilation will be applied to the input image instead. Default is False.

    Returns
    -------
    result : ndarray
        Erosion of input.
    """
    if structure is not None:
        raise NotImplementedError("'structure' is not yet supported.")
    if (size is None) == (footprint is None):
        raise ValueError("Exactly one of 'size' or 'footprint' must be provided or the function can't determine the working dimension!.")
    working_dim = footprint.ndim if footprint is not None else len(size)
    if footprint is None:
        footprint = ndimage.generate_binary_structure(working_dim, working_dim)
    if mask is not None and mask.shape != input.shape:
        raise ValueError("mask must have same shape as input.")
    if output is None:
        output = input.copy()
    elif output.shape != input.shape:
        raise ValueError("output must have same shape as input.")
    if input.ndim < working_dim:
        raise ValueError(f"Input must have at least {working_dim} dimensions given the footprint.")
    edge_mode_codes = {'reflect': 0, 'constant': 1, 'nearest': 2, 'mirror': 3, 'wrap': 4}
    if mode not in edge_mode_codes:
        raise ValueError("mode must be one of 'reflect','constant','nearest','mirror','wrap'")
    edge_mode_code = edge_mode_codes[mode]
    offsets = _get_offsets(footprint)
    # Make cval compatible with input dtype
    cval = np.array(cval, dtype=input.dtype).item()
    erosion = False if dilation else True
    batch = False

    if input.ndim > working_dim:
        # Reshape to (N, H, W) and process all slices in parallel
        original_shape = input.shape
        batch = True
        if working_dim == 2:
            H, W = original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-2]
            N = int(np.prod(leading_dims))
            input = input.reshape((N, H, W))
            output = output.reshape((N, H, W))
            mask = mask.reshape((N, H, W)) if mask is not None else None
        else:
            D, H, W = original_shape[-3], original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-3]
            N = int(np.prod(leading_dims))
            input = input.reshape((N, D, H, W))
            output = output.reshape((N, D, H, W))
            mask = mask.reshape((N, D, H, W)) if mask is not None else None

    i = 0
    changed = True
    while changed:
        changed = _scan_filter(input, output, mask, None, offsets, edge_mode_code, cval, erosion, working_dim, batch)
        i += 1
        if i >= iterations & iterations > 0:
            break
        if ~changed and i < iterations:
            print(f'Erosion ending prematurely: image no longer changed after {i} iterations.')
            break
    return output.reshape(original_shape) if output.ndim > working_dim else output

