import numpy as np
import scipy.ndimage as ndimage
from ._skeletonise import _skeletonize_internal_loop


def skeletonize(input, dim=2, inplace=False):
    """
    Compute the skeleton of the input image via thinning, using lee's algorithm.
    Support only 2D or 3D operation. But the input array can have arbitrary number of leading dimensions.
    The operation will be parallelised over the leading dimensions.
    The last 2 or 3 dimensions are treated as spatial dimensions: depth, height, and width.

    Uses Lee’s algorithm. Should produce almost identical result as to skimage.morphology.skeletonize for 3D images.
    Will produce different but still correct result for 2D images.

    Parameters
    ----------
    input : ndarray
        Array over which the grayscale erosion is to be computed.
    dim : int, optional
        The number of spatial dimension.
    inplace : bool, optional
        if true, perform the operation inplace.

    Returns
    -------
    skeleton : ndarray
        The binary thinned image.
    """
    if dim > input.ndim:
        raise ValueError('dim must be less than or equal to the number of input dimensions.')
    if dim < 2 or dim > 3:
        raise ValueError('dim must be 2 or 3.')
    if inplace:
        img = input
    else:
        img = input.astype(np.bool_, copy=True)

    original_shape = input.shape
    if input.ndim > dim:
        if dim == 2:
            # Reshape to (N, H, W) and process all slices in parallel
            H, W = original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-2]
            N = int(np.prod(leading_dims))
            img = img.reshape((N, H, W))
        else:
            D, H, W = original_shape[-3], original_shape[-2], original_shape[-1]
            leading_dims = original_shape[:-3]
            N = int(np.prod(leading_dims))
            img = img.reshape((N, D, H, W))

    _skeletonize_internal_loop(img, dim)

    return img

