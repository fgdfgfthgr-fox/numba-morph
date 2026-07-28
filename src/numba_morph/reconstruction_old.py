import numpy as np
import warnings
from numba import njit, prange, types
from .utils import _get_offsets_3d, _split_offsets_3d, _get_offsets_2d, _split_offsets_2d


# 2D core functions
@njit
def _scan_2d(mask, result, changed, offsets, reverse, erosion):
    """
    One raster scan pass for a 2D slice.
    """
    shape = mask.shape
    if not reverse:
        y_range = range(shape[0])
        x_range = range(shape[1])
    else:
        y_range = range(shape[0] - 1, -1, -1)
        x_range = range(shape[1] - 1, -1, -1)

    n_offsets = offsets.shape[0]
    for y in y_range:
        for x in x_range:
            current = result[y, x]

            if erosion:  # propagate minimum
                best = current
                for i in range(n_offsets):
                    dy = offsets[i, 0]
                    dx = offsets[i, 1]
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                        val = result[ny, nx]
                        if val < best:
                            best = val
                m = mask[y, x]
                if m > best:
                    best = m
                if best < current:
                    result[y, x] = best
                    changed = True
            else:       # propagate maximum
                best = current
                for i in range(n_offsets):
                    dy = offsets[i, 0]
                    dx = offsets[i, 1]
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                        val = result[ny, nx]
                        if val > best:
                            best = val
                m = mask[y, x]
                if m < best:
                    best = m
                if best > current:
                    result[y, x] = best
                    changed = True

    return result, changed


@njit
def _reconstruct_2d_slice(mask_slice, result_slice, forward_offsets, backward_offsets, erosion):
    """Iterative reconstruction on a single 2D slice until convergence."""
    changed = True
    while changed:
        changed = False
        result_slice, changed = _scan_2d(mask_slice, result_slice, changed,
                                         forward_offsets, False, erosion)
        result_slice, changed = _scan_2d(mask_slice, result_slice, changed,
                                         backward_offsets, True, erosion)


@njit(parallel=True)
def _reconstruct_2d_parallel(mask_flat, result_flat, forward_offsets, backward_offsets, erosion):
    """Parallel reconstruction over flattened leading dimensions."""
    n_slices = mask_flat.shape[0]
    for i in prange(n_slices):
        _reconstruct_2d_slice(mask_flat[i], result_flat[i],
                              forward_offsets, backward_offsets, erosion)


# 3D core functions
@njit
def _scan_3d(mask, result, changed, offsets, reverse, erosion):
    """
    One raster scan pass for a 3D volume.
    """
    shape = mask.shape
    if not reverse:
        z_range = range(shape[0])
        y_range = range(shape[1])
        x_range = range(shape[2])
    else:
        z_range = range(shape[0] - 1, -1, -1)
        y_range = range(shape[1] - 1, -1, -1)
        x_range = range(shape[2] - 1, -1, -1)

    n_offsets = offsets.shape[0]
    for z in z_range:
        for y in y_range:
            for x in x_range:
                current = result[z, y, x]

                if erosion:
                    best = current
                    for i in range(n_offsets):
                        dz = offsets[i, 0]
                        dy = offsets[i, 1]
                        dx = offsets[i, 2]
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx
                        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                            val = result[nz, ny, nx]
                            if val < best:
                                best = val
                    m = mask[z, y, x]
                    if m > best:
                        best = m
                    if best < current:
                        result[z, y, x] = best
                        changed = True
                else:
                    best = current
                    for i in range(n_offsets):
                        dz = offsets[i, 0]
                        dy = offsets[i, 1]
                        dx = offsets[i, 2]
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx
                        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                            val = result[nz, ny, nx]
                            if val > best:
                                best = val
                    m = mask[z, y, x]
                    if m < best:
                        best = m
                    if best > current:
                        result[z, y, x] = best
                        changed = True

    return result, changed


@njit
def _reconstruct_3d_slice(mask_slice, result_slice, forward_offsets, backward_offsets, erosion):
    """Iterative reconstruction on a single 3D volume until convergence."""
    changed = True
    while changed:
        changed = False
        result_slice, changed = _scan_3d(mask_slice, result_slice, changed,
                                         forward_offsets, False, erosion)
        result_slice, changed = _scan_3d(mask_slice, result_slice, changed,
                                         backward_offsets, True, erosion)


@njit(parallel=True)
def _reconstruct_3d_parallel(mask_flat, result_flat, forward_offsets, backward_offsets, erosion):
    """Parallel reconstruction over flattened leading dimensions."""
    n_slices = mask_flat.shape[0]
    for i in prange(n_slices):
        _reconstruct_3d_slice(mask_flat[i], result_flat[i],
                              forward_offsets, backward_offsets, erosion)


# Public 2D functions
def reconstruction_by_erosion_2d(mask, seed=None, dynamic=None, connectivity=4):
    """
    Geodesic reconstruction by erosion (propagation of minima) for 2D images.
    The last two axes are treated as height, and width.

    Parameters
    ----------
    mask : ndarray
        The mask image (lower bound). Last two dimensions are H, W.
    seed : ndarray, optional
        The seed image. If provided, it is used as the initial result.
    dynamic : scalar, optional
        If provided, the initial result is set to ``mask + dynamic``.
        Only one of `seed` or `dynamic` may be given.
    connectivity : {4, 8}
        Neighbourhood connectivity. Default is 4.

    Returns
    -------
    result : ndarray
        Reconstructed input.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if (seed is None) == (dynamic is None):
        raise ValueError("Exactly one of 'marker' or 'dynamic' must be provided.")
    if mask.ndim < 2:
        raise ValueError("mask must have at least 2 dimensions")
    if seed is not None and np.any(seed < mask):
        raise ValueError(
            "Intensity of seed image must be more than that "
            "of the mask image for reconstruction by erosion."
        )

    H, W = mask.shape[-2], mask.shape[-1]
    mask_flat = np.ascontiguousarray(mask.reshape(-1, H, W))

    if seed is not None:
        if seed.shape != mask.shape:
            raise ValueError("seed must have same shape as mask")
        result = seed
    else:
        result = mask + dynamic

    offsets = _get_offsets_2d(connectivity)
    forward_offsets, backward_offsets = _split_offsets_2d(offsets)

    if mask.ndim == 2:
        _reconstruct_2d_slice(mask, result, forward_offsets, backward_offsets, True)
        return result

    result_flat = np.ascontiguousarray(result.reshape(-1, H, W))

    _reconstruct_2d_parallel(mask_flat, result_flat, forward_offsets, backward_offsets, True)

    return result_flat.reshape(mask.shape)


def reconstruction_by_dilation_2d(mask, seed=None, dynamic=None, connectivity=4):
    """
    Geodesic reconstruction by dilation (propagation of maxima) for 2D images.
    The last two axes are treated as height, and width.

    Parameters
    ----------
    mask : ndarray
        The mask image (upper bound). Last two dimensions are H, W.
    seed : ndarray, optional
        The seed image. If provided, it is used as the initial result.
    dynamic : scalar, optional
        If provided, the initial result is set to ``mask - dynamic``.
        Only one of `seed` or `dynamic` may be given.
    connectivity : {4, 8}
        Neighbourhood connectivity. Default is 4.

    Returns
    -------
    result : ndarray
        Reconstructed input.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if (seed is None) == (dynamic is None):
        raise ValueError("Exactly one of 'seed' or 'dynamic' must be provided.")
    if mask.ndim < 2:
        raise ValueError("mask must have at least 2 dimensions")
    if seed is not None and np.any(seed > mask):
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )

    H, W = mask.shape[-2], mask.shape[-1]
    mask_flat = np.ascontiguousarray(mask.reshape(-1, H, W))

    if seed is not None:
        if seed.shape != mask.shape:
            raise ValueError("marker must have same shape as mask")
        result = seed
    else:
        if dynamic > mask.min() and np.issubdtype(mask.dtype, np.unsignedinteger):
            warnings.warn(
                f"dynamic ({dynamic}) exceeds mask.min() ({mask.min()}). "
                "Clipping negative values to 0.",
                RuntimeWarning
            )
            result = np.zeros_like(mask)
            np.subtract(mask, dynamic, out=result, where=(mask > dynamic))
        else:
            result = mask - dynamic

    offsets = _get_offsets_2d(connectivity)
    forward_offsets, backward_offsets = _split_offsets_2d(offsets)

    if mask.ndim == 2:
        _reconstruct_2d_slice(mask, result, forward_offsets, backward_offsets, False)
        return result

    result_flat = np.ascontiguousarray(result.reshape(-1, H, W))

    _reconstruct_2d_parallel(mask_flat, result_flat, forward_offsets, backward_offsets, False)

    return result_flat.reshape(mask.shape)


# Public 3D functions
def reconstruction_by_erosion_3d(mask, seed=None, dynamic=None, connectivity=26):
    """
    Geodesic reconstruction by erosion (propagation of minima) for 3D volumes.
    The last three axes are treated as depth, height, and width.

    Parameters
    ----------
    mask : ndarray
        The mask image (lower bound). Last three dimensions are D, H, W.
    seed : ndarray, optional
        The seed image. If provided, it is used as the initial result.
    dynamic : scalar, optional
        If provided, the initial result is set to ``mask + dynamic``.
        Only one of `seed` or `dynamic` may be given.
    connectivity : {6, 18, 26}
        Neighbourhood connectivity. Default is 26.

    Returns
    -------
    result : ndarray
        Reconstructed input.
    """
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")
    if (seed is None) == (dynamic is None):
        raise ValueError("Exactly one of 'seed' or 'dynamic' must be provided.")
    if mask.ndim < 3:
        raise ValueError("mask must have at least 3 dimensions")
    if seed is not None and np.any(seed < mask):
        raise ValueError(
            "Intensity of seed image must be more than that "
            "of the mask image for reconstruction by erosion."
        )

    D, H, W = mask.shape[-3], mask.shape[-2], mask.shape[-1]
    mask_flat = np.ascontiguousarray(mask.reshape(-1, D, H, W))

    if seed is not None:
        if seed.shape != mask.shape:
            raise ValueError("seed must have same shape as mask")
        result = seed
    else:
        result = mask + dynamic

    result_flat = np.ascontiguousarray(result.reshape(-1, D, H, W))

    offsets = _get_offsets_3d(connectivity)
    forward_offsets, backward_offsets = _split_offsets_3d(offsets)

    if mask.ndim == 3:
        _reconstruct_3d_slice(mask, result, forward_offsets, backward_offsets, True)
        return result

    _reconstruct_3d_parallel(mask_flat, result_flat, forward_offsets, backward_offsets, True)

    return result_flat.reshape(mask.shape)


def reconstruction_by_dilation_3d(mask, seed=None, dynamic=None, connectivity=26):
    """
    Geodesic reconstruction by dilation (propagation of maxima) for 3D volumes.
    The last three axes are treated as depth, height, and width.

    Parameters
    ----------
    mask : ndarray
        The mask image (upper bound). Last three dimensions are D, H, W.
    seed : ndarray, optional
        The seed image. If provided, it is used as the initial result.
    dynamic : scalar, optional
        If provided, the initial result is set to ``mask - dynamic``.
        Only one of `seed` or `dynamic` may be given.
    connectivity : {6, 18, 26}
        Neighbourhood connectivity. Default is 26.

    Returns
    -------
    result : ndarray
        Reconstructed input.
    """
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")
    if (seed is None) == (dynamic is None):
        raise ValueError("Exactly one of 'seed' or 'dynamic' must be provided.")
    if mask.ndim < 3:
        raise ValueError("mask must have at least 3 dimensions")
    if seed is not None and np.any(seed > mask):
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )

    D, H, W = mask.shape[-3], mask.shape[-2], mask.shape[-1]
    mask_flat = np.ascontiguousarray(mask.reshape(-1, D, H, W))

    if seed is not None:
        if seed.shape != mask.shape:
            raise ValueError("seed must have same shape as mask")
        result = seed
    else:
        if dynamic > mask.min() and np.issubdtype(mask.dtype, np.unsignedinteger):
            warnings.warn(
                f"dynamic ({dynamic}) exceeds mask.min() ({mask.min()}). "
                "Clipping negative values to 0.",
                RuntimeWarning
            )
            result = np.zeros_like(mask)
            np.subtract(mask, dynamic, out=result, where=(mask > dynamic))
        else:
            result = mask - dynamic

    offsets = _get_offsets_3d(connectivity)
    forward_offsets, backward_offsets = _split_offsets_3d(offsets)

    if mask.ndim == 3:
        _reconstruct_3d_slice(mask, result, forward_offsets, backward_offsets, False)
        return result

    result_flat = np.ascontiguousarray(result.reshape(-1, D, H, W))

    _reconstruct_3d_parallel(mask_flat, result_flat, forward_offsets, backward_offsets, False)

    return result_flat.reshape(mask.shape)

if __name__ == '__main__':
    test_mask = np.random.randint(10, 255, (16, 100, 100, 100), dtype=np.uint16)
    out = reconstruction_by_dilation_3d(test_mask, seed=None, dynamic=10, connectivity=26)