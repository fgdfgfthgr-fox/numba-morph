import numpy as np
from numba import njit, prange, stencil


def _get_offsets(structure):
    """
    Return relative offset coordinates for every ``True`` neighbour in the
    binary structure (footprint).  The centre element is excluded.

    Parameters
    ----------
    structure : array_like
        Boolean array (e.g. from ``scipy.ndimage.generate_binary_structure``).
        Must have odd size along every dimension so that a unique centre exists.

    Returns
    -------
    offsets : ndarray of shape (N, ndim), dtype=int32
        Each row ``(d_0, d_1, ..., d_{ndim-1})`` is an offset from the centre
        to a neighbour that is ``True`` in *structure*.
    """
    structure = np.asarray(structure, dtype=bool)
    if any(s % 2 == 0 for s in structure.shape):
        raise ValueError("structure must have odd size in every dimension.")

    # All indices where structure is True
    idx = np.argwhere(structure)                # shape (M, ndim)
    centre = np.array(structure.shape) // 2     # (ndim,)
    offsets = idx - centre

    # Remove the centre itself (offset == 0)
    offsets = offsets[np.any(offsets != 0, axis=1)]
    return offsets.astype(np.int32)


def _split_offsets(offsets):
    """
    Split offset vectors into *forward* and *backward* neighbours according
    to C‑order (raster scan) traversal: the first dimension that differs
    determines the half – negative → forward, positive → backward.

    Parameters
    ----------
    offsets : ndarray of shape (N, ndim), dtype=int
        Offset vectors as returned by `_get_offsets`.

    Returns
    -------
    forward : ndarray, shape (F, ndim), dtype=int32
        Offsets that appear *before* the centre in raster order.
    backward : ndarray, shape (B, ndim), dtype=int32
        Offsets that appear *after* the centre in raster order.
    """
    offsets = np.asarray(offsets, dtype=np.int32)
    if offsets.size == 0:
        return np.empty((0, offsets.shape[1]), dtype=np.int32), \
               np.empty((0, offsets.shape[1]), dtype=np.int32)

    # First dimension (column) that is non-zero
    first_nonzero = np.argmax(offsets != 0, axis=1)   # (N,)
    # Sign of that first non-zero element
    direction = offsets[np.arange(len(offsets)), first_nonzero]

    forward = offsets[direction < 0]
    backward = offsets[direction > 0]
    return forward, backward


@njit(inline="always", cache=True)
def _reflect_handle_2d(nh, nw, H, W):
    if nh < 0:
        nh = -nh - 1
    elif nh >= H:
        nh = 2 * H - nh - 1
    if nw < 0:
        nw = -nw - 1
    elif nw >= W:
        nw = 2 * W - nw - 1
    return nh, nw

@njit(inline="always", cache=True)
def _reflect_handle_3d(nd, nh, nw, D, H, W):
    if nd < 0:
        nd = -nd - 1
    elif nd >= D:
        nd = 2 * D - nd - 1
    if nh < 0:
        nh = -nh - 1
    elif nh >= H:
        nh = 2 * H - nh - 1
    if nw < 0:
        nw = -nw - 1
    elif nw >= W:
        nw = 2 * W - nw - 1
    return nd, nh, nw

@njit(inline="always", cache=True)
def _nearest_handle_2d(nh, nw, H, W):
    if nh < 0:
        nh = 0
    elif nh >= H:
        nh = H - 1
    if nw < 0:
        nw = 0
    elif nw >= W:
        nw = W - 1
    return nh, nw

@njit(inline="always", cache=True)
def _nearest_handle_3d(nd, nh, nw, D, H, W):
    if nd < 0:
        nd = 0
    elif nd >= D:
        nd = D - 1
    if nh < 0:
        nh = 0
    elif nh >= H:
        nh = H - 1
    if nw < 0:
        nw = 0
    elif nw >= W:
        nw = W - 1
    return nd, nh, nw

@njit(inline="always", cache=True)
def _mirror_handle_2d(nh, nw, H, W):
    if nh < 0 or nh >= H:
        p = 2 * H - 2
        nh = nh % p
        if nh >= H:
            nh = p - nh
    if nw < 0 or nw >= W:
        p = 2 * W - 2
        nw = nw % p
        if nw >= W:
            nw = p - nw
    return nh, nw

@njit(inline="always", cache=True)
def _mirror_handle_3d(nd, nh, nw, D, H, W):
    if nd < 0 or nd >= D:
        p = 2 * D - 2
        nd = nd % p
        if nd >= D:
            nd = p - nd
    if nh < 0 or nh >= H:
        p = 2 * H - 2
        nh = nh % p
        if nh >= H:
            nh = p - nh
    if nw < 0 or nw >= W:
        p = 2 * W - 2
        nw = nw % p
        if nw >= W:
            nw = p - nw
    return nd, nh, nw

@njit(inline="always", cache=True)
def _wrap_handle_2d(nh, nw, H, W):
    nh = nh % H
    nw = nw % W
    return nh, nw

@njit(inline="always", cache=True)
def _wrap_handle_3d(nd, nh, nw, D, H, W):
    nd = nd % D
    nh = nh % H
    nw = nw % W
    return nd, nh, nw


@njit(parallel=True, fastmath=True, cache=True)
def _scan_2d_filter(arr, out, mask, bound, min_val, max_val, offsets, mode_code, cval, erosion):
    """Approach erosion/dilation as a min/max filter."""
    H, W = arr.shape
    n_offsets = offsets.shape[0]
    changed = 0

    for i in prange(H * W):
        h = i // W
        w = i % W
        if mask is not None and mask[h, w] == 0:
            continue
        val = arr[h, w]
        if (val == min_val and erosion) or (val == max_val and not erosion):
            out[h, w] = val
            continue
        best = val

        for idx in range(n_offsets):
            nh = h + offsets[idx, 0]
            nw = w + offsets[idx, 1]

            # boundary handling
            if mode_code == 1 and (nh < 0 or nh >= H or nw < 0 or nw >= W):
                neighbor = cval
            else:
                if mode_code == 0:  # 'reflect'
                    nh, nw = _reflect_handle_2d(nh, nw, H, W)
                elif mode_code == 2:  # 'nearest'
                    nh, nw = _nearest_handle_2d(nh, nw, H, W)
                elif mode_code == 3:  # 'mirror'
                    nh, nw = _mirror_handle_2d(nh, nw, H, W)
                elif mode_code == 4:  # 'wrap'
                    nh, nw = _wrap_handle_2d(nh, nw, H, W)
                neighbor = arr[nh, nw]

            if not erosion:
                best = max(neighbor, best)
            else:
                best = min(neighbor, best)

        if bound is not None:
            if not erosion:
                best = min(best, bound[h,w])
            else:
                best = max(best, bound[h,w])
        if best != val:
            changed += 1
        out[h, w] = best
    return True if changed >= 1 else False


@njit(parallel=True, fastmath=True, cache=True)
def _scan_2d_filter_batch(arr, out, mask, bound, min_val, max_val, offsets, mode_code, cval, erosion):
    """Approach erosion/dilation as a min/max filter. This one has a leading dimension."""
    L, H, W = arr.shape
    n_offsets = offsets.shape[0]
    changed = 0

    for i in prange(L*H*W):
        l = i // (H * W)
        h = (i // W) % H
        w = i % W
        if mask is not None and mask[l, h, w] == 0:
            continue
        val = arr[l, h, w]
        if (val == min_val and erosion) or (val == max_val and not erosion):
            out[l, h, w] = val
            continue
        best = val

        for idx in range(n_offsets):
            nh = h + offsets[idx, 0]
            nw = w + offsets[idx, 1]

            # boundary handling
            if mode_code == 1 and (nh < 0 or nh >= H or nw < 0 or nw >= W):
                neighbor = cval
            else:
                if mode_code == 0:  # 'reflect'
                    nh, nw = _reflect_handle_2d(nh, nw, H, W)
                elif mode_code == 2:  # 'nearest'
                    nh, nw = _nearest_handle_2d(nh, nw, H, W)
                elif mode_code == 3:  # 'mirror'
                    nh, nw = _mirror_handle_2d(nh, nw, H, W)
                elif mode_code == 4:  # 'wrap'
                    nh, nw = _wrap_handle_2d(nh, nw, H, W)
                neighbor = arr[l, nh, nw]

            if not erosion:
                best = max(neighbor, best)
            else:
                best = min(neighbor, best)
        if bound is not None:
            if not erosion:
                best = min(best, bound[l,h,w])
            else:
                best = max(best, bound[l,h,w])
        if best != val:
            changed += 1
        out[l, h, w] = best
    return True if changed >= 1 else False


@njit(parallel=True, fastmath=True, cache=True)
def _scan_3d_filter(arr, out, mask, bound, min_val, max_val, offsets, mode_code, cval, erosion):
    D, H, W = arr.shape
    N = D * H * W
    n_offsets = offsets.shape[0]
    changed = 0
    for i in prange(N):
        d = i // (H * W)
        h = (i // W) % H
        w = i % W
        if mask is not None and mask[d, h, w] == 0:
            continue
        val = arr[d, h, w]
        if (val == min_val and erosion) or (val == max_val and not erosion):
            out[d, h, w] = val
            continue
        best = val

        for idx in range(n_offsets):
            nd = d + offsets[idx, 0]
            nh = h + offsets[idx, 1]
            nw = w + offsets[idx, 2]

            # boundary handling
            if mode_code == 1 and (nd < 0 or nd >=D or nh < 0 or nh >= H or nw < 0 or nw >= W):
                neighbor = cval
            else:
                if mode_code == 0:  # 'reflect'
                    nd, nh, nw = _reflect_handle_3d(nd, nh, nw, D, H, W)
                elif mode_code == 2:  # 'nearest'
                    nd, nh, nw = _nearest_handle_3d(nd, nh, nw, D, H, W)
                elif mode_code == 3:  # 'mirror'
                    nd, nh, nw = _mirror_handle_3d(nd, nh, nw, D, H, W)
                elif mode_code == 4:  # 'wrap'
                    nd, nh, nw = _wrap_handle_3d(nd, nh, nw, D, H, W)
                neighbor = arr[nd, nh, nw]

            if not erosion:
                best = max(neighbor, best)
            else:
                best = min(neighbor, best)
        if bound is not None:
            if not erosion:
                best = min(best, bound[d, h, w])
            else:
                best = max(best, bound[d, h, w])
        if best != val:
            changed += 1

        out[d, h, w] = best

    return True if changed >= 1 else False


@njit(parallel=True, fastmath=True, cache=True)
def _scan_3d_filter_batch(arr, out, mask, bound, min_val, max_val, offsets, mode_code, cval, erosion):
    L, D, H, W = arr.shape
    N = L * D * H * W
    n_offsets = offsets.shape[0]
    changed = 0

    for i in prange(N):
        spatial_size = D * H * W
        l = i // spatial_size
        rem = i % spatial_size
        d = rem // (H * W)
        h = (rem // W) % H
        w = rem % W
        if mask is not None and mask[l, d, h, w] == 0:
            continue
        val = arr[l, d, h, w]
        if (val == min_val and erosion) or (val == max_val and not erosion):
            out[l, d, h, w] = val
            continue
        best = val

        for idx in range(n_offsets):
            nd = d + offsets[idx, 0]
            nh = h + offsets[idx, 1]
            nw = w + offsets[idx, 2]

            if mode_code == 1 and (nd < 0 or nd >= D or nh < 0 or nh >= H or nw < 0 or nw >= W):
                neighbor = cval
            else:
                if mode_code == 0:  # 'reflect'
                    nd, nh, nw = _reflect_handle_3d(nd, nh, nw, D, H, W)
                elif mode_code == 2:  # 'nearest'
                    nd, nh, nw = _nearest_handle_3d(nd, nh, nw, D, H, W)
                elif mode_code == 3:  # 'mirror'
                    nd, nh, nw = _mirror_handle_3d(nd, nh, nw, D, H, W)
                elif mode_code == 4:  # 'wrap'
                    nd, nh, nw = _wrap_handle_3d(nd, nh, nw, D, H, W)
                neighbor = arr[l, nd, nh, nw]

            if not erosion:
                best = max(neighbor, best)
            else:
                best = min(neighbor, best)
        if bound is not None:
            if not erosion:
                best = min(best, bound[l, d, h, w])
            else:
                best = max(best, bound[l, d, h, w])
        if best != val:
            changed += 1

        out[l, d, h, w] = best

    return True if changed >= 1 else False


@njit(cache=True)
def _scan_2d_raster(arr, mask, bound, min_val, max_val, offsets, reverse, mode_code, cval, erosion):
    """
    Approach erosion/dilation using raster scan.
    Slower but memory efficient. Modifies arr in place.
    """
    H, W = arr.shape
    changed = False
    if not reverse:
        h_range = range(H)
        w_range = range(W)
    else:
        h_range = range(H - 1, -1, -1)
        w_range = range(W - 1, -1, -1)

    n_offsets = offsets.shape[0]

    for h in h_range:
        for w in w_range:
            if mask is not None and mask[h, w] == 0:
                continue
            current = arr[h, w]
            if (current == min_val and erosion) or (current == max_val and not erosion):
                continue
            best = current

            for idx in range(n_offsets):
                nh = h + offsets[idx, 0]
                nw = w + offsets[idx, 1]
                if mode_code == 1 and (nh < 0 or nh >= H or nw < 0 or nw >= W):
                    neighbor = cval
                else:
                    if mode_code == 0:  # 'reflect'
                        nh, nw = _reflect_handle_2d(nh, nw, H, W)
                    elif mode_code == 2:  # 'nearest'
                        nh, nw = _nearest_handle_2d(nh, nw, H, W)
                    elif mode_code == 3:  # 'mirror'
                        nh, nw = _mirror_handle_2d(nh, nw, H, W)
                    elif mode_code == 4:  # 'wrap'
                        nh, nw = _wrap_handle_2d(nh, nw, H, W)
                    neighbor = arr[nh, nw]

                if not erosion:
                    best = max(neighbor, best)
                else:
                    best = min(neighbor, best)

            m = bound[h, w]
            if erosion:
                if m > best:
                    best = m
            else:
                if m < best:
                    best = m

            if best != current:
                arr[h, w] = best
                changed = True

    return changed


@njit(parallel=True, fastmath=True, cache=True)
def _scan_2d_raster_batch(arr, mask, bound, min_val, max_val, offsets, reverse, mode_code, cval, erosion):
    """
    Raster scan for 3D arrays with a leading dimension (L, H, W).
    Parallelized over the leading dimension. Modifies arr in place.
    """
    L, H, W = arr.shape
    changed = False
    if not reverse:
        h_range = range(H)
        w_range = range(W)
    else:
        h_range = range(H - 1, -1, -1)
        w_range = range(W - 1, -1, -1)
    n_offsets = offsets.shape[0]
    changed_per_l = np.zeros(L, dtype=np.bool_)

    for l in prange(L):
        changed_l = False
        for h in h_range:
            for w in w_range:
                if mask is not None and mask[l, h, w] == 0:
                    continue
                current = arr[l, h, w]
                if (current == min_val and erosion) or (current == max_val and not erosion):
                    continue
                best = current

                for idx in range(n_offsets):
                    nh = h + offsets[idx, 0]
                    nw = w + offsets[idx, 1]
                    if mode_code == 1 and (nh < 0 or nh >= H or nw < 0 or nw >= W):
                        neighbor = cval
                    else:
                        if mode_code == 0:  # 'reflect'
                            nh, nw = _reflect_handle_2d(nh, nw, H, W)
                        elif mode_code == 2:  # 'nearest'
                            nh, nw = _nearest_handle_2d(nh, nw, H, W)
                        elif mode_code == 3:  # 'mirror'
                            nh, nw = _mirror_handle_2d(nh, nw, H, W)
                        elif mode_code == 4:  # 'wrap'
                            nh, nw = _wrap_handle_2d(nh, nw, H, W)
                        neighbor = arr[l, nh, nw]

                    if not erosion:
                        best = max(neighbor, best)
                    else:
                        best = min(neighbor, best)

                m = bound[l, h, w]
                if erosion:
                    if m > best:
                        best = m
                else:
                    if m < best:
                        best = m

                if best != current:
                    arr[l, h, w] = best
                    changed_l = True

        changed_per_l[l] = changed_l

    if np.any(changed_per_l):
        changed = True
    return changed


@njit(cache=True)
def _scan_3d_raster(arr, mask, bound, min_val, max_val, offsets, reverse, mode_code, cval, erosion):
    """
    Approach erosion/dilation using raster scan for 3D arrays.
    Slower but memory efficient. Modifies arr in place.
    """
    D, H, W = arr.shape
    changed = False
    if not reverse:
        d_range = range(D)
        h_range = range(H)
        w_range = range(W)
    else:
        d_range = range(D - 1, -1, -1)
        h_range = range(H - 1, -1, -1)
        w_range = range(W - 1, -1, -1)

    n_offsets = offsets.shape[0]

    for d in d_range:
        for h in h_range:
            for w in w_range:
                if mask is not None and mask[d, h, w] == 0:
                    continue
                current = arr[d, h, w]
                if (current == min_val and erosion) or (current == max_val and not erosion):
                    continue
                best = current

                for idx in range(n_offsets):
                    nd = d + offsets[idx, 0]
                    nh = h + offsets[idx, 1]
                    nw = w + offsets[idx, 2]

                    # boundary handling
                    if mode_code == 1 and (nd < 0 or nd >= D or nh < 0 or nh >= H or nw < 0 or nw >= W):
                        neighbor = cval
                    else:
                        if mode_code == 0:  # 'reflect'
                            nd, nh, nw = _reflect_handle_3d(nd, nh, nw, D, H, W)
                        elif mode_code == 2:  # 'nearest'
                            nd, nh, nw = _nearest_handle_3d(nd, nh, nw, D, H, W)
                        elif mode_code == 3:  # 'mirror'
                            nd, nh, nw = _mirror_handle_3d(nd, nh, nw, D, H, W)
                        elif mode_code == 4:  # 'wrap'
                            nd, nh, nw = _wrap_handle_3d(nd, nh, nw, D, H, W)
                        neighbor = arr[nd, nh, nw]

                    if not erosion:
                        best = max(neighbor, best)
                    else:
                        best = min(neighbor, best)

                m = bound[d, h, w]
                if erosion:
                    if m > best:
                        best = m
                else:
                    if m < best:
                        best = m

                if best != current:
                    arr[d, h, w] = best
                    changed = True

    return changed


@njit(parallel=True, fastmath=True, cache=True)
def _scan_3d_raster_batch(arr, mask, bound, min_val, max_val, offsets, reverse, mode_code, cval, erosion):
    """
    Raster scan for 4D arrays with a leading dimension (L, D, H, W).
    Parallelized over the leading dimension. Modifies arr in place.
    """
    L, D, H, W = arr.shape
    changed = False
    if not reverse:
        d_range = range(D)
        h_range = range(H)
        w_range = range(W)
    else:
        d_range = range(D - 1, -1, -1)
        h_range = range(H - 1, -1, -1)
        w_range = range(W - 1, -1, -1)
    n_offsets = offsets.shape[0]
    changed_per_l = np.zeros(L, dtype=np.bool_)

    for l in prange(L):
        changed_l = False
        for d in d_range:
            for h in h_range:
                for w in w_range:
                    if mask is not None and mask[l, d, h, w] == 0:
                        continue
                    current = arr[l, d, h, w]
                    if (current == min_val and erosion) or (current == max_val and not erosion):
                        continue
                    best = current

                    for idx in range(n_offsets):
                        nd = d + offsets[idx, 0]
                        nh = h + offsets[idx, 1]
                        nw = w + offsets[idx, 2]

                        # boundary handling
                        if mode_code == 1 and (nd < 0 or nd >= D or nh < 0 or nh >= H or nw < 0 or nw >= W):
                            neighbor = cval
                        else:
                            if mode_code == 0:  # 'reflect'
                                nd, nh, nw = _reflect_handle_3d(nd, nh, nw, D, H, W)
                            elif mode_code == 2:  # 'nearest'
                                nd, nh, nw = _nearest_handle_3d(nd, nh, nw, D, H, W)
                            elif mode_code == 3:  # 'mirror'
                                nd, nh, nw = _mirror_handle_3d(nd, nh, nw, D, H, W)
                            elif mode_code == 4:  # 'wrap'
                                nd, nh, nw = _wrap_handle_3d(nd, nh, nw, D, H, W)
                            neighbor = arr[l, nd, nh, nw]

                        if not erosion:
                            best = max(neighbor, best)
                        else:
                            best = min(neighbor, best)

                    m = bound[l, d, h, w]
                    if erosion:
                        if m > best:
                            best = m
                    else:
                        if m < best:
                            best = m

                    if best != current:
                        arr[l, d, h, w] = best
                        changed_l = True

        changed_per_l[l] = changed_l

    if np.any(changed_per_l):
        changed = True
    return changed


def _scan_filter(arr, output, mask, bound, offsets, edge_mode_code, cval, erosion, working_dim, batch):
    if arr.dtype == np.float16:
        NotImplementedError("numba-morph doesn't support float16 input! This is a limit of numba.")
    if batch:
        scan_function = _scan_2d_filter_batch if working_dim == 2 else _scan_3d_filter_batch
    else:
        scan_function = _scan_2d_filter if working_dim == 2 else _scan_3d_filter
    if np.issubdtype(arr.dtype, np.floating):
        info = np.finfo(arr.dtype)
        min_val, max_val = info.min, info.max
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        min_val, max_val = info.min, info.max
    else:
        min_val, max_val = 0, 1

    changed = scan_function(arr, output, mask, bound, min_val, max_val, offsets, edge_mode_code, cval, erosion)
    return changed

def _scan_raster(arr, mask, bound, offsets, edge_mode_code, cval, erosion, working_dim, batch):
    if arr.dtype == np.float16:
        NotImplementedError("numba-morph doesn't support float16 input! This is a limit of numba.")
    if bound is None:
        ValueError("Raster scan algorithm can't be used in non-reconstruction context!")

    if batch:
        scan_function = _scan_2d_raster_batch if working_dim == 2 else _scan_3d_raster_batch
    else:
        scan_function = _scan_2d_raster if working_dim == 2 else _scan_3d_raster

    if np.issubdtype(arr.dtype, np.floating):
        info = np.finfo(arr.dtype)
        min_val, max_val = info.min, info.max
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        min_val, max_val = info.min, info.max
    else:
        min_val, max_val = 0, 1

    forward_offsets, backward_offsets = _split_offsets(offsets)
    # Forward pass (raster order)
    changed_fwd = scan_function(
        arr, mask, bound, min_val, max_val, forward_offsets, False,
        edge_mode_code, cval, erosion)

    # Backward pass (reverse raster order)
    changed_bwd = scan_function(
        arr, mask, bound, min_val, max_val, backward_offsets, True,
        edge_mode_code, cval, erosion)

    changed = changed_fwd or changed_bwd
    return changed

@stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def _dilate_26(window, mask):
    val = window[0, 0, 0]
    best = val
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                best = max(best, window[dz, dy, dx])
    if mask is not None:
        best = min(best, mask[0, 0, 0])
    return best

@njit(parallel=True)
def _dilate_26_batch(arr, mask):
    changed = 0
    out = _dilate_26(arr, mask)
    return out, changed
