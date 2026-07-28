import numpy as np
import os

from ._chamfer import _chamfer, _generate_offsets
from .utils import choose_algorithm

def distance_transform_cdt(input, dtype=np.uint16, num_bands=None, weights=(3,4), speed='auto'):
    """
    Perform Chamfer Distance Transform with parallelisation over the largest spatial axis.
    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.

    Whether this function choose 2D or 3D operation is determined by the length of 'weights'.

    Usually slower than scipy.ndimage.distance_transform_cdt unless the image is very large.
    But uses around 50% less memory.

    Parameters
    ----------
    input : ndarray
        Binary image; foreground > 0, background = 0.
    dtype : numpy dtype
        Data type for the distance map. If it's the same as input's dtype, then it become an in-place op.
        Default is np.uint16.
    num_bands : int, optional
        Number of parallel chunks. Default is cpu_count() - 1.
    weights : tuple of ints
        Weight for face and edge (and maybe corner) neighbours. Default is default (3,4) (Borgefors).
         (1,1) to replicate chessboard and (1,2) to replicate taxicab.
         For 3D, it's (3,4,5), (1,1,1) or (1,2,3).
    speed : str or bool
        If True, will use a (usually) faster, multi-threaded algorithm.
        If False, will use a slower, single-threaded algorithm.
        Default to "auto", which uses the shape of the input and number of available cpu-core to choose the algorithm.
        Auto will prioritise faster algorithm unless the single-threaded one is faster.

    Returns
    -------
    result : ndarray
        Distance transformed input.
    """
    working_dim = len(weights)
    if working_dim < 2 or working_dim > 3:
        raise ValueError(f"'weights' must have at least 2 and at most 3 elements. Currently: {working_dim}.")
    if input.ndim < working_dim:
        raise ValueError(f"Input must have at least {working_dim} dimensions given the weights.")
    if speed == 'auto':
        speed = choose_algorithm(input, working_dim, 2097152)
    causal, anti_causal = _generate_offsets(weights)
    input = input.astype(dtype, copy=False)
    max_val = np.iinfo(dtype).max
    input[input>0] = max_val
    batch = False

    if working_dim == 2:
        H, W = input.shape[-2], input.shape[-1]
        size_of_largest_dim = H if H >= W else W
        chunk_dim = -2 if H >= W else -1
    else:
        D, H, W = input.shape[-3], input.shape[-2], input.shape[-1]
        dims = {'D': (D, -3), 'H': (H, -2), 'X': (W, -1)}
        largest_name, (size_of_largest_dim, chunk_dim) = max(dims.items(), key=lambda kv: kv[1][0])

    if num_bands is None:
        num_bands = max(1, os.cpu_count() - 1)
        num_bands = min(num_bands, size_of_largest_dim//2)

    original_shape = input.shape
    if input.ndim > working_dim:
        batch = True
        if working_dim == 2:
            leading_dims = original_shape[:-2]
            N = int(np.prod(leading_dims))
            input = input.reshape((N, H, W))
        else:
            leading_dims = original_shape[:-3]
            N = int(np.prod(leading_dims))
            input = input.reshape((N, D, H, W))

    input = _chamfer(input, max_val, num_bands, causal, anti_causal, working_dim, speed, size_of_largest_dim, chunk_dim, batch)

    return input.reshape(original_shape) if input.ndim > working_dim else input
