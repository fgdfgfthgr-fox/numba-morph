import warnings
import numpy as np
from .utils import choose_algorithm, safe_add
from ._scan import _get_offsets, _scan_filter, _scan_raster
from ._propagation import _propagate


def reconstruction(mask, seed, inplace=False, method='dilation',
                   footprint=None, edge_mode='reflect', cval=0, speed='auto'):
    """
    Geodesic reconstruction for ndarray.
    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
    Whether to choose 2D or 3D operation is determined by the footprint.

    Parameters
    ----------
    mask : ndarray
        The mask image (lower or upper bound).
    seed : ndarray
        The marker image. Used as the initial result.
    inplace : bool
        If true, will perform the reconstruction in-place on seed. Otherwise, a new array will be created. Defaults to False.
    method : str, {'erosion', 'dilation'}
        In dilation (or erosion), the seed image is dilated (or eroded) until limited by the mask image.
        For dilation, each seed value must be less than or equal to the corresponding mask value;
        for erosion, the reverse is true.
    footprint : ndarray
        The neighborhood expressed as an n-D array of 1’s and 0’s.
        i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
    edge_mode : str, {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how the array borders are handled. default is 'reflect'.
    cval : int
        The value when mode is equal to 'constant'. Default is 0.
    speed : str or bool
        If True, will use a (usually) faster, multi-threaded algorithm that requires an intermediate tensor.
        If False, will use a slower, single-threaded algorithm that modifies the array in-place and saves memory.
        Default to "auto", which uses the shape of the input and number of available cpu-core to choose the algorithm.
        Auto will prioritise faster algorithm unless the single-threaded one is faster.

    Returns
    -------
    result : ndarray
        Reconstructed input.
    """
    if footprint is None:
        raise ValueError("A footprint must be provided or the function can't determine the working dimension!")
    else:
        working_dim = footprint.ndim
        if working_dim < 2 or working_dim > 3:
            raise ValueError(f"'footprint' must have at least 2 and at most 3 dimensions. Currently: {footprint.ndim}.")
    if mask.ndim < working_dim:
        raise ValueError(f"mask must have at least {working_dim} dimensions given the footprint.")
    if seed.shape != mask.shape:
        raise ValueError("seed must have same shape as mask")
    if np.any(seed < mask) and method == 'erosion':
        raise ValueError(
            "Intensity of seed image must be more than that "
            "of the mask image for reconstruction by erosion."
        )
    elif np.any(seed > mask) and method == 'dilation':
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )
    if inplace:
        result = seed
    else:
        result = seed.copy()
    edge_mode_codes = {'reflect': 0, 'constant': 1, 'nearest': 2, 'mirror': 3, 'wrap': 4}
    if edge_mode not in edge_mode_codes:
        raise ValueError("mode must be one of 'reflect','constant','nearest','mirror','wrap'")
    edge_mode_code = edge_mode_codes[edge_mode]
    method_codes = {'erosion': True, 'dilation': False}
    if method not in method_codes:
        raise ValueError("method must be either 'dilation' or 'erosion'.")
    method_code = method_codes[method]
    if speed == 'auto':
        speed = choose_algorithm(result, working_dim)
    offsets = _get_offsets(footprint)
    # Make cval compatible with input dtype
    cval = np.array(cval, dtype=result.dtype).item()

    changed = True

    batch = False
    if result.ndim > working_dim:
        # Reshape to (N, H, W) and process all slices in parallel
        original_shape = result.shape
        batch = True
        if working_dim == 2:
            H, W = original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-2]
            N = int(np.prod(leading_dims))
            result = result.reshape((N, H, W))
            mask = mask.reshape((N, H, W))
        else:
            D, H, W = original_shape[-3], original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-3]
            N = int(np.prod(leading_dims))
            result = result.reshape((N, D, H, W))
            mask = mask.reshape((N, D, H, W))

    # First Scan using erosion or dilation
    if speed:
        _scan_filter(result, result, None, mask, offsets, edge_mode_code, cval, method_code, working_dim, batch)
    else:
        _scan_raster(result, None, mask, offsets, edge_mode_code, cval, method_code, working_dim, batch)

    # Heapq priority queue
    _propagate(result, mask, offsets, method_code, working_dim, batch)

    return result.reshape(original_shape) if result.ndim > working_dim else result

