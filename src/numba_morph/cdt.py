import warnings
import numpy as np
from .utils import set_positive_to_val
from ._chamfer import _chamfer, _generate_offsets

def distance_transform_cdt(input, dtype=np.uint16, weights=(3,4), output=None):
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
        Data type for the distance map if 'output' is not provided.
        Default is np.uint16.
    weights : tuple of ints
        Weight for face and edge (and maybe corner) neighbours. Default is default (3,4) (Borgefors).
         (1,1) to replicate chessboard and (1,2) to replicate taxicab.
         For 3D, it's (3,4,5), (1,1,1) or (1,2,3).
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed. By default, a copy of the input array is created.

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
    causal, anti_causal = _generate_offsets(weights)
    if output is None:
        output = input.astype(dtype, copy=True)
    if output.dtype == np.bool_:
        raise ValueError("bool data type for output is not useful for distance transform.")
    elif output.dtype == np.uint8 or output.dtype == np.int8:
        warnings.warn(
            'uint8 or int8 data type for output '
            'may not have enough capacity for distance transform. '
            'Be aware of that.',
            stacklevel=2,
        )
    max_val = np.iinfo(output.dtype).max if np.issubdtype(output.dtype, np.integer) else np.finfo(output.dtype).max
    set_positive_to_val(output, max_val) # In-place, saves memory compare to numpy.where
    batch = False

    original_shape = output.shape
    if output.ndim > working_dim:
        batch = True
        if working_dim == 2:
            H, W = output.shape[-2], output.shape[-1]
            leading_dims = original_shape[:-2]
            N = int(np.prod(leading_dims))
            output = output.reshape((N, H, W))
        else:
            D, H, W = output.shape[-3], output.shape[-2], output.shape[-1]
            leading_dims = original_shape[:-3]
            N = int(np.prod(leading_dims))
            output = output.reshape((N, D, H, W))

    output = _chamfer(output, max_val, causal, anti_causal, working_dim, batch)

    return output.reshape(original_shape) if output.ndim > working_dim else output
