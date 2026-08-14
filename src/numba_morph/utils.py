import numba
from numba import njit, prange
import numpy as np

def generate_sphere_structure(radius):
    """
    Generate a binary footprint that approximates an ellipse or ellipsoid with specified radii along each axis.
    
    Parameters
    ----------
    radius : array-like of floats
        The radius along each axis. The length of the tuple/array determines the number of dimensions.
    
    Returns
    -------
    footprint : ndarray
        A boolean array of shape (2*ceil(radius[0])+1, 2*ceil(radius[1])+1, ...).
        Elements are True if their center lies within the ellipsoid defined by the radii.
    """
    radii = np.asarray(radius)
    dims = radii.size

    # Compute half-sizes for each dimension (ceil of radius)
    sizes = np.ceil(radii).astype(int)

    # Generate coordinate ranges for each dimension
    coords = [np.arange(-sizes[i], sizes[i] + 1) for i in range(dims)]

    # Create meshgrid (with 'ij' indexing to match dimension order)
    grid = np.meshgrid(*coords, indexing='ij')

    # Compute the sum of squared normalized coordinates for non-zero radii
    sum_sq = np.zeros(grid[0].shape, dtype=float)
    for i, coord in enumerate(grid):
        if radii[i] > 0:
            sum_sq += (coord / radii[i]) ** 2

    # Points inside (or on) the ellipsoid
    footprint = sum_sq <= 1.0
    return footprint


def choose_algorithm(arr, algo_dim, size_threshold=1024):
    """Choose whether to use raster or filter algorithm based on cpu cores and array size."""
    available_cpu_core = numba.config.NUMBA_DEFAULT_NUM_THREADS
    # The raster algorithm will always be faster if only 1 cpu core available.
    if available_cpu_core == 1:
        return False
    # Array needs to be large enough for filter algorithm to be faster.
    if arr.size <= size_threshold:
        return False
    else:
        if arr.ndim == algo_dim:
            # No leading dim means no parallel for raster scan.
            return True
        else:
            leading_dims = arr.shape[:-algo_dim]
            N = int(np.prod(leading_dims))
            if N >= (available_cpu_core//2) and N >= 1:
                # Fully parallel over leading dim. Raster scan has less overhead and better memory property than filter.
                # Choose over half the cpu cores because hyper-threading.
                return False
            else:
                # Leading dim small and do not utilise the cpu cores.
                return True

def safe_add(mask, dynamic):
    """
    Add scalar `dynamic` to array `mask` with clamping to the valid range
    of the array's data type for integer types.

    Parameters
    ----------
    mask : np.ndarray
        Input array (any numeric dtype).
    dynamic : int or float
        Scalar value to add.

    Returns
    -------
    np.ndarray
        Result with same shape and dtype as `mask`. For integer dtypes,
        values are clipped to [dtype.min, dtype.max] to avoid wrap‑around.
    """
    # Floating-point: standard addition (no clamping)
    if np.issubdtype(mask.dtype, np.floating):
        return mask + dynamic

    # Integer types: use a safe intermediate (int64) block‑wise
    info = np.iinfo(mask.dtype)
    min_val, max_val = info.min, info.max

    result = np.empty_like(mask)
    total = mask.size

    chunk_size = 4096

    # Work on the flattened array for simplicity
    flat_mask = mask.ravel()
    flat_result = result.ravel()

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        block = flat_mask[start:end]  # view, no copy

        # Cast to int64, add the scalar, clip to the original dtype limits,
        # then cast back to the original dtype.
        temp = block.astype(np.int64) + dynamic
        clipped = np.clip(temp, min_val, max_val)
        flat_result[start:end] = clipped.astype(mask.dtype)

    return result

@njit(cache=True, parallel=True)
def set_positive_to_val(arr, val):
    for i in prange(arr.size):
        if arr.flat[i] > 0:
            arr.flat[i] = val