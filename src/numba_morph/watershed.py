import numpy as np
import warnings
from ._watershed import _marker_controlled_watershed
from ._scan import _get_offsets

def watershed(input, markers=None, footprint=None, mask=None):
    """
        Find watershed basins in an image flooded from given markers.

        Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
        The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.
        Whether to choose 2D or 3D operation is determined by the footprint.

        Parameters
        ----------
        input : array_like
            Data array where the lowest value points are labeled first.
        markers : array_like
            The desired number of basins, or an array marking the basins with the values to be assigned in the label
            matrix. Zero means not a marker.
            Unlike skimage, markers here needs to be explicitly passed.
        footprint : array of ints, optional
            The neighborhood expressed as an n-D array of 1’s and 0’s.
            i.e. a 3x3 square for 2D, a 3x3x3 cube for 3D.
        mask : array_like, optional
            Array of same shape as image. Only points at which mask == True will be labeled.

        Returns
        -------
        result : ndarray
            A labeled matrix of the same type and shape as markers.
    """
    if markers is None:
        raise ValueError("markers needs to be explicitly passed")
    if footprint is None:
        raise ValueError("A footprint must be provided or the function can't determine the working dimension!")
    else:
        working_dim = footprint.ndim
        if working_dim < 2 or working_dim > 3:
            raise ValueError(f"'footprint' must have at least 2 and at most 3 dimensions. Currently: {footprint.ndim}.")
    if input.ndim < working_dim:
        raise ValueError(f"input must have at least {working_dim} dimensions given the footprint.")
    if input.shape != markers.shape:
        raise ValueError("input must have same shape as markers")
    batch = False
    offsets = _get_offsets(footprint)

    if input.ndim > working_dim:
        # Reshape to (N, H, W) and process all slices in parallel
        original_shape = input.shape
        batch = True
        if working_dim == 2:
            H, W = original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-2]
            N = int(np.prod(leading_dims))
            input = input.reshape((N, H, W))
            markers = markers.reshape((N, H, W))
            mask = mask.reshape((N, H, W)) if mask is not None else None
        else:
            D, H, W = original_shape[-3], original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-3]
            N = int(np.prod(leading_dims))
            input = input.reshape((N, D, H, W))
            markers = markers.reshape((N, D, H, W))
            mask = mask.reshape((N, D, H, W)) if mask is not None else None

    markers = _marker_controlled_watershed(input, markers, mask=mask, offsets=offsets, batch=batch, working_dim=working_dim)
    return markers