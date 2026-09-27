import math
import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id
from ._propagation import (
    _propagate_2d,
    _propagate_3d,
    _propagate_2d_batched,
    _propagate_3d_batched,
)

_MIN_TILED_SIZE_2D = 1024 * 1024
_MIN_TILED_SIZE_3D = 128 * 128 * 128

# ----------------------------------------------------------------------
# Dirty-tile bookkeeping (pure Python; n_tiles is small so this is cheap)
#
# A tile must be re-solved iff the values of its *halo* changed since the
# last time it was solved. The halo is the tile's own interior plus the
# 1-voxel border contributed by adjacent tiles, so it is enough to mark a
# tile dirty when it, or any of its 8 (2D) / 26 (3D) neighbours, changed.
# ----------------------------------------------------------------------
def _compute_dirty_2d(changed, n_tiles_h, n_tiles_w):
    changed_2d = changed.reshape(n_tiles_h, n_tiles_w)
    dirty = np.zeros((n_tiles_h, n_tiles_w), dtype=bool)
    for ti, tj in zip(*np.where(changed_2d)):
        i0 = ti - 1 if ti > 0 else 0
        i1 = ti + 2 if ti + 2 < n_tiles_h else n_tiles_h
        j0 = tj - 1 if tj > 0 else 0
        j1 = tj + 2 if tj + 2 < n_tiles_w else n_tiles_w
        dirty[i0:i1, j0:j1] = True
    return dirty

def _compute_dirty_2d_batched(changed, n_tiles_h, n_tiles_w):
    B = changed.shape[0]
    changed_3d = changed.reshape(B, n_tiles_h, n_tiles_w)
    dirty = np.zeros((B, n_tiles_h, n_tiles_w), dtype=bool)
    for b in range(B):
        for ti, tj in zip(*np.where(changed_3d[b])):
            i0 = ti - 1 if ti > 0 else 0
            i1 = ti + 2 if ti + 2 < n_tiles_h else n_tiles_h
            j0 = tj - 1 if tj > 0 else 0
            j1 = tj + 2 if tj + 2 < n_tiles_w else n_tiles_w
            dirty[b, i0:i1, j0:j1] = True
    return dirty

def _compute_dirty_3d(changed, n_tiles_d, n_tiles_h, n_tiles_w):
    changed_3d = changed.reshape(n_tiles_d, n_tiles_h, n_tiles_w)
    dirty = np.zeros((n_tiles_d, n_tiles_h, n_tiles_w), dtype=bool)
    for ti, tj, tk in zip(*np.where(changed_3d)):
        i0 = ti - 1 if ti > 0 else 0
        i1 = ti + 2 if ti + 2 < n_tiles_d else n_tiles_d
        j0 = tj - 1 if tj > 0 else 0
        j1 = tj + 2 if tj + 2 < n_tiles_h else n_tiles_h
        k0 = tk - 1 if tk > 0 else 0
        k1 = tk + 2 if tk + 2 < n_tiles_w else n_tiles_w
        dirty[i0:i1, j0:j1, k0:k1] = True
    return dirty


def _compute_dirty_3d_batched(changed, n_tiles_d, n_tiles_h, n_tiles_w):
    B = changed.shape[0]
    changed_4d = changed.reshape(B, n_tiles_d, n_tiles_h, n_tiles_w)
    dirty = np.zeros((B, n_tiles_d, n_tiles_h, n_tiles_w), dtype=bool)
    for b in range(B):
        for ti, tj, tk in zip(*np.where(changed_4d[b])):
            i0 = ti - 1 if ti > 0 else 0
            i1 = ti + 2 if ti + 2 < n_tiles_d else n_tiles_d
            j0 = tj - 1 if tj > 0 else 0
            j1 = tj + 2 if tj + 2 < n_tiles_h else n_tiles_h
            k0 = tk - 1 if tk > 0 else 0
            k1 = tk + 2 if tk + 2 < n_tiles_w else n_tiles_w
            dirty[b, i0:i1, j0:j1, k0:k1] = True
    return dirty

@njit(parallel=True, fastmath=True, cache=True)
def _tiled_sweep_2d(src, dst, mask, offsets, mode_code, cval, erosion, dtype,
                    tile_h, tile_w, n_tiles_w, n_tiles,
                    dirty_idx, scratch_marker, scratch_mask):
    H = src.shape[0]
    W = src.shape[1]
    n_dirty = dirty_idx.shape[0]
    changed = np.zeros(n_tiles, dtype=np.bool_)

    for i in prange(n_dirty):
        t = dirty_idx[i]
        ti = t // n_tiles_w
        tj = t % n_tiles_w

        h0 = ti * tile_h
        h1 = min(h0 + tile_h, H)
        w0 = tj * tile_w
        w1 = min(w0 + tile_w, W)

        hh0 = max(h0 - 1, 0)
        hh1 = min(h1 + 1, H)
        ww0 = max(w0 - 1, 0)
        ww1 = min(w1 + 1, W)
        th = hh1 - hh0
        tw = ww1 - ww0

        tid = get_thread_id()
        local_marker = scratch_marker[tid, :th, :tw]
        local_mask = scratch_mask[tid, :th, :tw]

        local_marker[:, :] = src[hh0:hh1, ww0:ww1]
        local_mask[:, :] = mask[hh0:hh1, ww0:ww1]

        _propagate_2d(local_marker, local_mask, offsets,
                      mode_code, cval, erosion, dtype)

        new_slice = local_marker[h0 - hh0:h1 - hh0, w0 - ww0:w1 - ww0]
        old_slice = src[h0:h1, w0:w1]
        changed[t] = not np.array_equal(new_slice, old_slice)
        dst[h0:h1, w0:w1] = new_slice

    return changed


@njit(parallel=True, fastmath=True, cache=True)
def _tiled_sweep_2d_batched(src, dst, mask, offsets, mode_code, cval, erosion, dtype,
                            tile_h, tile_w, n_tiles_w, n_tiles,
                            dirty_b, dirty_t, scratch_marker, scratch_mask):
    B = src.shape[0]
    H = src.shape[1]
    W = src.shape[2]
    n_dirty = dirty_b.shape[0]
    changed = np.zeros((B, n_tiles), dtype=np.bool_)

    for i in prange(n_dirty):
        b = dirty_b[i]
        t = dirty_t[i]
        ti = t // n_tiles_w
        tj = t % n_tiles_w

        h0 = ti * tile_h
        h1 = min(h0 + tile_h, H)
        w0 = tj * tile_w
        w1 = min(w0 + tile_w, W)

        hh0 = max(h0 - 1, 0)
        hh1 = min(h1 + 1, H)
        ww0 = max(w0 - 1, 0)
        ww1 = min(w1 + 1, W)
        th = hh1 - hh0
        tw = ww1 - ww0

        tid = get_thread_id()
        local_marker = scratch_marker[tid, :th, :tw]
        local_mask = scratch_mask[tid, :th, :tw]

        local_marker[:, :] = src[b, hh0:hh1, ww0:ww1]
        local_mask[:, :] = mask[b, hh0:hh1, ww0:ww1]

        _propagate_2d(local_marker, local_mask, offsets,
                      mode_code, cval, erosion, dtype)

        new_slice = local_marker[h0 - hh0:h1 - hh0, w0 - ww0:w1 - ww0]
        old_slice = src[b, h0:h1, w0:w1]
        changed[b, t] = not np.array_equal(new_slice, old_slice)
        dst[b, h0:h1, w0:w1] = new_slice

    return changed


# ----------------------------------------------------------------------
# 3D sweeps
# ----------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _tiled_sweep_3d(src, dst, mask, offsets, mode_code, cval, erosion, dtype,
                    tile_d, tile_h, tile_w, n_tiles_h, n_tiles_w, n_tiles,
                    dirty_idx, scratch_marker, scratch_mask):
    D = src.shape[0]
    H = src.shape[1]
    W = src.shape[2]
    n_dirty = dirty_idx.shape[0]
    changed = np.zeros(n_tiles, dtype=np.bool_)

    for i in prange(n_dirty):
        t = dirty_idx[i]
        ti = t // (n_tiles_h * n_tiles_w)
        rem = t % (n_tiles_h * n_tiles_w)
        tj = rem // n_tiles_w
        tk = rem % n_tiles_w

        d0 = ti * tile_d
        d1 = min(d0 + tile_d, D)
        h0 = tj * tile_h
        h1 = min(h0 + tile_h, H)
        w0 = tk * tile_w
        w1 = min(w0 + tile_w, W)

        dd0 = max(d0 - 1, 0)
        dd1 = min(d1 + 1, D)
        hh0 = max(h0 - 1, 0)
        hh1 = min(h1 + 1, H)
        ww0 = max(w0 - 1, 0)
        ww1 = min(w1 + 1, W)
        td = dd1 - dd0
        th = hh1 - hh0
        tw = ww1 - ww0

        tid = get_thread_id()
        local_marker = scratch_marker[tid, :td, :th, :tw]
        local_mask = scratch_mask[tid, :td, :th, :tw]

        local_marker[:, :, :] = src[dd0:dd1, hh0:hh1, ww0:ww1]
        local_mask[:, :, :] = mask[dd0:dd1, hh0:hh1, ww0:ww1]

        _propagate_3d(local_marker, local_mask, offsets,
                      mode_code, cval, erosion, dtype)

        new_slice = local_marker[d0 - dd0:d1 - dd0, h0 - hh0:h1 - hh0, w0 - ww0:w1 - ww0]
        old_slice = src[d0:d1, h0:h1, w0:w1]
        changed[t] = not np.array_equal(new_slice, old_slice)
        dst[d0:d1, h0:h1, w0:w1] = new_slice

    return changed


@njit(parallel=True, fastmath=True, cache=True)
def _tiled_sweep_3d_batched(src, dst, mask, offsets, mode_code, cval, erosion, dtype,
                            tile_d, tile_h, tile_w, n_tiles_h, n_tiles_w, n_tiles,
                            dirty_b, dirty_t, scratch_marker, scratch_mask):
    B = src.shape[0]
    D = src.shape[1]
    H = src.shape[2]
    W = src.shape[3]
    n_dirty = dirty_b.shape[0]
    changed = np.zeros((B, n_tiles), dtype=np.bool_)

    for i in prange(n_dirty):
        b = dirty_b[i]
        t = dirty_t[i]
        ti = t // (n_tiles_h * n_tiles_w)
        rem = t % (n_tiles_h * n_tiles_w)
        tj = rem // n_tiles_w
        tk = rem % n_tiles_w

        d0 = ti * tile_d
        d1 = min(d0 + tile_d, D)
        h0 = tj * tile_h
        h1 = min(h0 + tile_h, H)
        w0 = tk * tile_w
        w1 = min(w0 + tile_w, W)

        dd0 = max(d0 - 1, 0)
        dd1 = min(d1 + 1, D)
        hh0 = max(h0 - 1, 0)
        hh1 = min(h1 + 1, H)
        ww0 = max(w0 - 1, 0)
        ww1 = min(w1 + 1, W)
        td = dd1 - dd0
        th = hh1 - hh0
        tw = ww1 - ww0

        tid = get_thread_id()
        local_marker = scratch_marker[tid, :td, :th, :tw]
        local_mask = scratch_mask[tid, :td, :th, :tw]

        local_marker[:, :, :] = src[b, dd0:dd1, hh0:hh1, ww0:ww1]
        local_mask[:, :, :] = mask[b, dd0:dd1, hh0:hh1, ww0:ww1]

        _propagate_3d(local_marker, local_mask, offsets,
                      mode_code, cval, erosion, dtype)

        new_slice = local_marker[d0 - dd0:d1 - dd0, h0 - hh0:h1 - hh0, w0 - ww0:w1 - ww0]
        old_slice = src[b, d0:d1, h0:h1, w0:w1]
        changed[b, t] = not np.array_equal(new_slice, old_slice)
        dst[b, d0:d1, h0:h1, w0:w1] = new_slice

    return changed


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------
def _queue_2d_tiled(marker, mask, offsets, mode_code, cval, erosion, dtype):
    H, W = marker.shape

    n_threads = max(get_num_threads(), 1)
    target_tiles = 2 * n_threads
    tile_size = int(math.sqrt((H * W) / max(target_tiles, 1)))

    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    n_tiles_h = (H + tile_h - 1) // tile_h
    n_tiles_w = (W + tile_w - 1) // tile_w
    n_tiles = n_tiles_h * n_tiles_w

    max_th = min(tile_h + 2, H)
    max_tw = min(tile_w + 2, W)
    scratch_marker = np.empty((n_threads, max_th, max_tw), dtype=dtype)
    scratch_mask = np.empty((n_threads, max_th, max_tw), dtype=mask.dtype)

    buf = np.empty_like(marker)
    src, dst = marker, buf

    dirty = np.ones((n_tiles_h, n_tiles_w), dtype=bool)

    while True:
        np.copyto(dst, src)
        dirty_idx = np.flatnonzero(dirty.ravel()).astype(np.int64)
        changed = _tiled_sweep_2d(
            src, dst, mask, offsets, mode_code, cval, erosion, dtype,
            tile_h, tile_w, n_tiles_w, n_tiles,
            dirty_idx, scratch_marker, scratch_mask,
        )
        src, dst = dst, src
        if not changed.any():
            break
        dirty = _compute_dirty_2d(changed, n_tiles_h, n_tiles_w)

    if src is not marker:
        np.copyto(marker, src)


def _queue_2d_tiled_batched(marker, mask, offsets, mode_code, cval, erosion, dtype):
    B, H, W = marker.shape

    n_threads = max(get_num_threads(), 1)
    target_tiles = 2 * n_threads
    tile_size = int(math.sqrt((H * W) / max(target_tiles, 1)))

    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    n_tiles_h = (H + tile_h - 1) // tile_h
    n_tiles_w = (W + tile_w - 1) // tile_w
    n_tiles = n_tiles_h * n_tiles_w

    max_th = min(tile_h + 2, H)
    max_tw = min(tile_w + 2, W)
    scratch_marker = np.empty((n_threads, max_th, max_tw), dtype=dtype)
    scratch_mask = np.empty((n_threads, max_th, max_tw), dtype=mask.dtype)

    buf = np.empty_like(marker)
    src, dst = marker, buf

    dirty = np.ones((B, n_tiles_h, n_tiles_w), dtype=bool)

    while True:
        np.copyto(dst, src)
        flat = np.flatnonzero(dirty.ravel()).astype(np.int64)
        dirty_b = flat // n_tiles
        dirty_t = flat % n_tiles
        changed = _tiled_sweep_2d_batched(
            src, dst, mask, offsets, mode_code, cval, erosion, dtype,
            tile_h, tile_w, n_tiles_w, n_tiles,
            dirty_b, dirty_t, scratch_marker, scratch_mask,
        )
        src, dst = dst, src
        if not changed.any():
            break
        dirty = _compute_dirty_2d_batched(changed, n_tiles_h, n_tiles_w)

    if src is not marker:
        np.copyto(marker, src)


def _queue_3d_tiled(marker, mask, offsets, mode_code, cval, erosion, dtype):
    D, H, W = marker.shape

    n_threads = max(get_num_threads(), 1)
    target_tiles = 2 * n_threads
    tile_size = int((D * H * W / max(target_tiles, 1)) ** (1.0 / 3.0))

    tile_d = min(tile_size, D)
    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    n_tiles_d = (D + tile_d - 1) // tile_d
    n_tiles_h = (H + tile_h - 1) // tile_h
    n_tiles_w = (W + tile_w - 1) // tile_w
    n_tiles = n_tiles_d * n_tiles_h * n_tiles_w

    max_td = min(tile_d + 2, D)
    max_th = min(tile_h + 2, H)
    max_tw = min(tile_w + 2, W)
    scratch_marker = np.empty((n_threads, max_td, max_th, max_tw), dtype=dtype)
    scratch_mask = np.empty((n_threads, max_td, max_th, max_tw), dtype=mask.dtype)

    buf = np.empty_like(marker)
    src, dst = marker, buf

    dirty = np.ones((n_tiles_d, n_tiles_h, n_tiles_w), dtype=bool)

    while True:
        np.copyto(dst, src)
        dirty_idx = np.flatnonzero(dirty.ravel())
        changed = _tiled_sweep_3d(
            src, dst, mask, offsets, mode_code, cval, erosion, dtype,
            tile_d, tile_h, tile_w, n_tiles_h, n_tiles_w, n_tiles,
            dirty_idx, scratch_marker, scratch_mask,
        )
        src, dst = dst, src
        if not changed.any():
            break
        dirty = _compute_dirty_3d(changed, n_tiles_d, n_tiles_h, n_tiles_w)

    if src is not marker:
        np.copyto(marker, src)


def _queue_3d_tiled_batched(marker, mask, offsets, mode_code, cval, erosion, dtype):
    B, D, H, W = marker.shape

    n_threads = max(get_num_threads(), 1)
    target_tiles = 2 * n_threads
    tile_size = int((D * H * W / max(target_tiles, 1)) ** (1.0 / 3.0))

    tile_d = min(tile_size, D)
    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    n_tiles_d = (D + tile_d - 1) // tile_d
    n_tiles_h = (H + tile_h - 1) // tile_h
    n_tiles_w = (W + tile_w - 1) // tile_w
    n_tiles = n_tiles_d * n_tiles_h * n_tiles_w

    max_td = min(tile_d + 2, D)
    max_th = min(tile_h + 2, H)
    max_tw = min(tile_w + 2, W)
    scratch_marker = np.empty((n_threads, max_td, max_th, max_tw), dtype=dtype)
    scratch_mask = np.empty((n_threads, max_td, max_th, max_tw), dtype=mask.dtype)

    buf = np.empty_like(marker)
    src, dst = marker, buf

    dirty = np.ones((B, n_tiles_d, n_tiles_h, n_tiles_w), dtype=bool)

    while True:
        np.copyto(dst, src)
        flat = np.flatnonzero(dirty.ravel())
        dirty_b = flat // n_tiles
        dirty_t = flat % n_tiles
        changed = _tiled_sweep_3d_batched(
            src, dst, mask, offsets, mode_code, cval, erosion, dtype,
            tile_d, tile_h, tile_w, n_tiles_h, n_tiles_w, n_tiles,
            dirty_b, dirty_t, scratch_marker, scratch_mask,
        )
        src, dst = dst, src
        if not changed.any():
            break
        dirty = _compute_dirty_3d_batched(changed, n_tiles_d, n_tiles_h, n_tiles_w)

    if src is not marker:
        np.copyto(marker, src)


def _propagate(marker, mask, offsets, mode_code, cval, erosion,
               working_dim, batch):
    dtype = marker.dtype
    if dtype == np.float16:
        raise NotImplementedError(
            "numba-morph doesn't support float16 input! This is a limit of numba.")

    if batch:
        if working_dim == 2:
            if marker.shape[1] * marker.shape[2] < _MIN_TILED_SIZE_2D:
                _propagate_2d_batched(marker, mask, offsets,
                                      mode_code, cval, erosion, dtype.type)
            else:
                _queue_2d_tiled_batched(marker, mask, offsets,
                                        mode_code, cval, erosion, dtype.type)
        else:
            if marker.shape[1] * marker.shape[2] * marker.shape[3] < _MIN_TILED_SIZE_3D:
                _propagate_3d_batched(marker, mask, offsets,
                                      mode_code, cval, erosion, dtype.type)
            else:
                _queue_3d_tiled_batched(marker, mask, offsets,
                                        mode_code, cval, erosion, dtype.type)
        return

    if working_dim == 2:
        if marker.size < _MIN_TILED_SIZE_2D:
            _propagate_2d(marker, mask, offsets,
                          mode_code, cval, erosion, dtype.type)
        else:
            _queue_2d_tiled(marker, mask, offsets,
                            mode_code, cval, erosion, dtype.type)
    else:
        if marker.size < _MIN_TILED_SIZE_3D:
            _propagate_3d(marker, mask, offsets,
                          mode_code, cval, erosion, dtype.type)
        else:
            _queue_3d_tiled(marker, mask, offsets,
                            mode_code, cval, erosion, dtype.type)