import numpy as np
from numba import njit, prange


def _generate_offsets(weights):
    """
    Generate causal and anti-causal offsets for Chamfer masks.
    weights : tuple of ints
        For 2D: (face_weight, edge_weight)
        For 3D: (face_weight, edge_weight, corner_weight)
    Returns two arrays of shape (N, ndim+1) with columns [offset_0, ..., offset_{ndim-1}, weight].
    """
    ndim = len(weights)
    causal = []
    anti_causal = []

    if ndim == 2:
        # 2D offsets: (dy, dx)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                manhattan = abs(dy) + abs(dx)
                if manhattan > ndim:          # max Manhattan distance is 2
                    continue
                w = weights[manhattan - 1]
                # causal: first non‑zero component is negative
                if dy < 0 or (dy == 0 and dx < 0):
                    causal.append((dy, dx, w))
                if dy > 0 or (dy == 0 and dx > 0):
                    anti_causal.append((dy, dx, w))

    elif ndim == 3:
        # 3D offsets: (dz, dy, dx)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    manhattan = abs(dz) + abs(dy) + abs(dx)
                    if manhattan > ndim:       # max Manhattan distance is 3
                        continue
                    w = weights[manhattan - 1]
                    if dz < 0 or (dz == 0 and dy < 0) or (dz == 0 and dy == 0 and dx < 0):
                        causal.append((dz, dy, dx, w))
                    if dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0):
                        anti_causal.append((dz, dy, dx, w))
    else:
        raise ValueError("Only 2D or 3D are supported")

    return np.array(causal, dtype=np.int32), np.array(anti_causal, dtype=np.int32)


@njit(cache=True)
def _apply_pass_2d(input, offsets,
                   y_start, y_end, y_step,
                   x_start, x_end, x_step,
                   H, W, max_val):
    changed = False
    for y in range(y_start, y_end, y_step):
        for x in range(x_start, x_end, x_step):
            if input[y, x] == 0:
                continue
            cur = input[y, x]
            new_val = cur
            for dy, dx, w in offsets:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neigh = input[ny, nx]
                    if neigh != max_val:
                        cand = neigh + w
                        if cand < new_val:
                            new_val = cand
            if new_val < cur:
                input[y, x] = new_val
                changed = True
    return changed


@njit(parallel=True, cache=True)
def _apply_pass_2d_batch(input, offsets,
                         y_start, y_end, y_step,
                         x_start, x_end, x_step,
                         L, H, W, max_val):
    changed = 0
    for l in prange(L):
        for y in range(y_start, y_end, y_step):
            for x in range(x_start, x_end, x_step):
                if input[l, y, x] == 0:
                    continue
                cur = input[l, y, x]
                new_val = cur
                for dy, dx, w in offsets:
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        neigh = input[l, ny, nx]
                        if neigh != max_val:
                            cand = neigh + w
                            if cand < new_val:
                                new_val = cand
                if new_val < cur:
                    input[l, y, x] = new_val
                    changed += 1
    return True if changed > 0 else False


@njit(cache=True)
def _apply_pass_3d(input, offsets,
                   z_start, z_end, z_step,
                   y_start, y_end, y_step,
                   x_start, x_end, x_step,
                   D, H, W, max_val):
    changed = False
    for z in range(z_start, z_end, z_step):
        for y in range(y_start, y_end, y_step):
            for x in range(x_start, x_end, x_step):
                if input[z, y, x] == 0:
                    continue
                cur = input[z, y, x]
                new_val = cur
                for dz, dy, dx, w in offsets:
                    nz = z + dz
                    ny = y + dy
                    nx = x + dx
                    if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                        neigh = input[nz, ny, nx]
                        if neigh != max_val:
                            cand = neigh + w
                            if cand < new_val:
                                new_val = cand
                if new_val < cur:
                    input[z, y, x] = new_val
                    changed = True
    return changed


@njit(parallel=True, cache=True)
def _apply_pass_3d_batch(input, offsets,
                         z_start, z_end, z_step,
                         y_start, y_end, y_step,
                         x_start, x_end, x_step,
                         L, D, H, W, max_val):
    changed = 0
    for l in prange(L):
        for z in range(z_start, z_end, z_step):
            for y in range(y_start, y_end, y_step):
                for x in range(x_start, x_end, x_step):
                    if input[l, z, y, x] == 0:
                        continue
                    cur = input[l, z, y, x]
                    new_val = cur
                    for dz, dy, dx, w in offsets:
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx
                        if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                            neigh = input[l, nz, ny, nx]
                            if neigh != max_val:
                                cand = neigh + w
                                if cand < new_val:
                                    new_val = cand
                    if new_val < cur:
                        input[l, z, y, x] = new_val
                        changed += 1
    return True if changed > 0 else False


@njit(parallel=True, fastmath=True, cache=True)
def _chamfer_2d_chunk(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim):
    H, W = input.shape
    chunk_size = (size_of_largest_dim // num_bands) + 1

    for it in range(size_of_largest_dim):
        changed = np.zeros(num_bands, dtype=np.bool_)

        for band in prange(num_bands):
            start = band * chunk_size
            end = min(start + chunk_size, size_of_largest_dim)

            # forward pass
            if _apply_pass_2d(input, causal,
                              start, end, 1,  # y: start -> end-1
                              0, W, 1,  # x: 0 -> W-1
                              H, W, max_val):
                changed[band] = True

            # backward pass
            if _apply_pass_2d(input, anti_causal,
                             end - 1, start - 1, -1,  # y: end-1 down to start
                             W - 1, -1, -1,  # x: W-1 down to 0
                              H, W, max_val):
                changed[band] = True

        if not np.any(changed):
            break


@njit(parallel=True, fastmath=True, cache=True)
def _chamfer_2d_chunk_batch(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim):
    L, H, W = input.shape
    chunk_size = (size_of_largest_dim // num_bands) + 1

    for it in range(size_of_largest_dim):
        changed = np.zeros(num_bands, dtype=np.bool_)

        for band in prange(num_bands):
            start = band * chunk_size
            end = min(start + chunk_size, size_of_largest_dim)

            # forward pass
            if _apply_pass_2d_batch(input, causal,
                                    start, end, 1,  # y: start -> end-1
                                    0, W, 1,  # x: 0 -> W-1
                                    L, H, W, max_val):
                changed[band] = True

            # backward pass
            if _apply_pass_2d_batch(input, anti_causal,
                                    end - 1, start - 1, -1,  # y: end-1 down to start
                                    W - 1, -1, -1,  # x: W-1 down to 0
                                    L, H, W, max_val):
                changed[band] = True

        if not np.any(changed):
            break


@njit(parallel=True, fastmath=True, cache=True)
def _chamfer_3d_chunk(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim):

    D, H, W = input.shape
    chunk_size = (size_of_largest_dim // num_bands) + 1

    for it in range(size_of_largest_dim):
        changed = np.zeros(num_bands, dtype=np.bool_)

        for band in prange(num_bands):
            start = band * chunk_size
            end = min(start + chunk_size, size_of_largest_dim)

            # forward pass
            if _apply_pass_3d(input, causal,
                              start, end, 1,  # y: start -> end-1
                              0, H, 1,
                              0, W, 1,  # x: 0 -> W-1
                              D, H, W, max_val):
                changed[band] = True

            # backward pass
            if _apply_pass_3d(input, anti_causal,
                              end - 1, start - 1, -1,  # y: end-1 down to start
                              H - 1, -1, -1,
                              W - 1, -1, -1,  # x: W-1 down to 0
                              D, H, W, max_val):
                changed[band] = True

        if not np.any(changed):
            break


@njit(parallel=True, fastmath=True, cache=True)
def _chamfer_3d_chunk_batch(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim):
    L, D, H, W = input.shape
    chunk_size = (size_of_largest_dim // num_bands) + 1

    for it in range(size_of_largest_dim):
        changed = np.zeros(num_bands, dtype=np.bool_)

        for band in prange(num_bands):
            start = band * chunk_size
            end = min(start + chunk_size, size_of_largest_dim)

            # forward pass
            if _apply_pass_3d_batch(input, causal,
                                    start, end, 1,
                                    0, H, 1,
                                    0, W, 1,  # x: 0 -> W-1
                                    L, D, H, W, max_val):
                changed[band] = True

            # backward pass
            if _apply_pass_3d_batch(input, anti_causal,
                                    end - 1, start - 1, -1,
                                    H - 1, -1, -1,
                                    W - 1, -1, -1,  # x: W-1 down to 0
                                    L, D, H, W, max_val):
                changed[band] = True

        if not np.any(changed):
            break


def reorder_offset_list(offset_list, spatial_perm):
    new_list = []
    for item in offset_list:
        spatial_offsets = item[:-1]  # all but the last element (weight)
        weight = item[-1]
        # Reorder spatial offsets according to spatial_perm
        new_spatial = tuple(spatial_offsets[i] for i in spatial_perm)
        new_list.append((*new_spatial, weight))
    return new_list


def _make_ranges(shape):
    """Return (fwd_ranges, bwd_ranges) for a given spatial shape."""
    fwd = tuple((0, d, 1) for d in shape)
    bwd = tuple((d - 1, -1, -1) for d in shape)
    return fwd, bwd


# ---------- Dispatcher for a single forward/backward pass ----------
def _run_pass(input, causal, anti_causal, ranges_fwd, ranges_bwd, ndim, batch, max_val):
    """
    Call the appropriate _apply_pass_* function with the given ranges.
    'ranges_fwd' and 'ranges_bwd' are tuples of (start, end, step) per axis.
    """
    if ndim == 2:
        if batch:
            func = _apply_pass_2d_batch
            # ranges: (y_range, x_range)
            (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_fwd
            func(input, causal, y_s, y_e, y_st, x_s, x_e, x_st, input.shape[0], input.shape[1], input.shape[2], max_val)
            (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_bwd
            func(input, anti_causal, y_s, y_e, y_st, x_s, x_e, x_st, input.shape[0], input.shape[1], input.shape[2], max_val)
        else:
            func = _apply_pass_2d
            (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_fwd
            func(input, causal, y_s, y_e, y_st, x_s, x_e, x_st, input.shape[0], input.shape[1], max_val)
            (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_bwd
            func(input, anti_causal, y_s, y_e, y_st, x_s, x_e, x_st, input.shape[0], input.shape[1], max_val)
    else:  # ndim == 3
        if batch:
            func = _apply_pass_3d_batch
            (z_s, z_e, z_st), (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_fwd
            func(input, causal, z_s, z_e, z_st, y_s, y_e, y_st, x_s, x_e, x_st,
                 input.shape[0], input.shape[1], input.shape[2], input.shape[3], max_val)
            (z_s, z_e, z_st), (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_bwd
            func(input, anti_causal, z_s, z_e, z_st, y_s, y_e, y_st, x_s, x_e, x_st,
                 input.shape[0], input.shape[1], input.shape[2], input.shape[3], max_val)
        else:
            func = _apply_pass_3d
            (z_s, z_e, z_st), (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_fwd
            func(input, causal, z_s, z_e, z_st, y_s, y_e, y_st, x_s, x_e, x_st,
                 input.shape[0], input.shape[1], input.shape[2], max_val)
            (z_s, z_e, z_st), (y_s, y_e, y_st), (x_s, x_e, x_st) = ranges_bwd
            func(input, anti_causal, z_s, z_e, z_st, y_s, y_e, y_st, x_s, x_e, x_st,
                 input.shape[0], input.shape[1], input.shape[2], max_val)


# ---------- Dispatcher for the chunked iterative version ----------
def _run_chunked(input, max_val, num_bands, causal, anti_causal, ndim, batch, size_of_largest_dim):
    """Call the appropriate _chamfer_*_chunk* function."""
    if ndim == 2:
        if batch:
            _chamfer_2d_chunk_batch(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim)
        else:
            _chamfer_2d_chunk(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim)
    else:  # ndim == 3
        if batch:
            _chamfer_3d_chunk_batch(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim)
        else:
            _chamfer_3d_chunk(input, max_val, num_bands, causal, anti_causal, size_of_largest_dim)


# ---------- Helper: permute for chunking ----------
def _permute_for_chunk(input, causal, anti_causal, working_dim, chunk_dim, batch):
    """
    Transpose the input so that the chunked axis becomes the first spatial axis,
    and reorder the offsets accordingly. Returns (transposed_input, new_causal, new_anti_causal, perm, inv_perm).
    """
    spatial_index = working_dim + chunk_dim          # absolute axis (0-based among spatial)
    if batch:
        total_axes = working_dim + 1
        abs_chunk_axis = spatial_index + 1           # batch is axis 0
        perm = [0, abs_chunk_axis] + [i for i in range(1, total_axes) if i != abs_chunk_axis]
        spatial_perm = [i - 1 for i in perm[1:]]     # spatial order after permutation
    else:
        perm = [spatial_index] + [i for i in range(working_dim) if i != spatial_index]
        spatial_perm = perm

    new_causal = reorder_offset_list(causal, spatial_perm)
    new_anti_causal = reorder_offset_list(anti_causal, spatial_perm)
    return np.transpose(input, perm), new_causal, new_anti_causal, perm, np.argsort(perm)


# ---------- Refactored _chamfer ----------
def _chamfer(input, max_val, num_bands, causal, anti_causal, working_dim,
             chunk, size_of_largest_dim, chunk_dim, batch):
    """
    Compute Chamfer distance transform.
    """
    if chunk:
        # Move the chunked axis to the front (after batch if present)
        input, causal, anti_causal, perm, inv_perm = _permute_for_chunk(
            input, causal, anti_causal, working_dim, chunk_dim, batch
        )
        # Run the chunked algorithm (the chunk function assumes the first spatial axis is the chunked one)
        _run_chunked(input, max_val, num_bands, causal, anti_causal, working_dim, batch, size_of_largest_dim)
        # Restore original axis order
        input = np.transpose(input, inv_perm)
    else:
        # No chunking: simply run one forward and one backward pass over all axes
        shape = input.shape[1:] if batch else input.shape   # spatial shape
        fwd_ranges, bwd_ranges = _make_ranges(shape)
        _run_pass(input, causal, anti_causal, fwd_ranges, bwd_ranges, working_dim, batch, max_val)

    return input