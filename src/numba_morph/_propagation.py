import numpy as np
from numba import njit, prange
from heapq import heappush, heappop

# TODO: Once Python 3.14 is popular, uses heapify_max instead.

@njit(fastmath=True)
def _propagate_2d(marker, mask, offsets, erosion, dtype):
    H, W = marker.shape
    max_val = np.max(marker) if erosion else np.max(mask)
    heap = [(dtype(0), np.uint16(0), np.uint16(0))]
    heap.pop()

    for h in range(H):
        for w in range(W):
            v = marker[h, w]
            can_propagate = False
            for dh, dw in offsets:
                nh = h + dh
                nw = w + dw
                if 0 <= nh < H and 0 <= nw < W:
                    v_idx = marker[nh, nw]
                    mask_idx = mask[nh, nw]
                    if v_idx > v and v_idx > mask_idx and erosion:
                        can_propagate = True
                        break
                    elif v_idx < v and v_idx < mask_idx and not erosion:
                        can_propagate = True
                        break
            if can_propagate:
                heappush(heap, (dtype(max_val - v), np.uint16(h), np.uint16(w)))

    while heap:
        priority, h, w = heappop(heap)
        v = max_val - priority
        if marker[h, w] != v:
            continue  # stale entry

        for dh, dw in offsets:
            nh = h + dh
            nw = w + dw
            if 0 <= nh < H and 0 <= nw < W:
                v_idx = marker[nh, nw]
                mask_idx = mask[nh, nw]
                if v_idx > v and v_idx > mask_idx and erosion:
                    new_v = max(v, mask_idx)
                    marker[nh, nw] = new_v
                    heappush(heap, (dtype(max_val - new_v), np.uint16(nh), np.uint16(nw)))
                elif v_idx < v and v_idx < mask_idx and not erosion:
                    new_v = min(v, mask_idx)
                    marker[nh, nw] = new_v
                    heappush(heap, (dtype(max_val - new_v), np.uint16(nh), np.uint16(nw)))


@njit(parallel=True, fastmath=True)
def _propagate_2d_batched(marker, mask, offsets, erosion, dtype):
    for l in prange(marker.shape[0]):
        _propagate_2d(marker[l], mask[l], offsets, erosion, dtype)


@njit(fastmath=True)
def _propagate_3d(marker, mask, offsets, erosion, dtype):
    D, H, W = marker.shape
    max_val = np.max(marker) if erosion else np.max(mask)
    heap = [(dtype(0), np.uint16(0), np.uint16(0), np.uint16(0))]
    heap.pop()

    for d in range(D):
        for h in range(H):
            for w in range(W):
                v = marker[d, h, w]
                can_propagate = False
                for dd, dh, dw in offsets:
                    nd = d + dd
                    nh = h + dh
                    nw = w + dw
                    if 0 <= nd < D and 0 <= nh < H and 0 <= nw < W:
                        v_idx = marker[nd, nh, nw]
                        mask_idx = mask[nd, nh, nw]
                        if v_idx > v and v_idx > mask_idx and erosion:
                            can_propagate = True
                            break
                        elif v_idx < v and v_idx < mask_idx and not erosion:
                            can_propagate = True
                            break
                if can_propagate:
                    heappush(heap, (dtype(max_val - v), np.uint16(d), np.uint16(h), np.uint16(w)))

    while heap:
        priority, d, h, w = heappop(heap)
        v = max_val - priority
        if marker[d, h, w] != v:
            continue  # stale entry

        for dd, dh, dw in offsets:
            nd = d + dd
            nh = h + dh
            nw = w + dw
            if 0 <= nd < D and 0 <= nh < H and 0 <= nw < W:
                v_idx = marker[nd, nh, nw]
                mask_idx = mask[nd, nh, nw]
                if v_idx > v and v_idx > mask_idx and erosion:
                    new_v = max(v, mask_idx)
                    marker[nd, nh, nw] = new_v
                    heappush(heap, (dtype(max_val - new_v), np.uint16(nd), np.uint16(nh), np.uint16(nw)))
                elif v_idx < v and v_idx < mask_idx and not erosion:
                    new_v = min(v, mask_idx)
                    marker[nd, nh, nw] = new_v
                    heappush(heap, (dtype(max_val - new_v), np.uint16(nd), np.uint16(nh), np.uint16(nw)))


@njit(parallel=True, fastmath=True)
def _propagate_3d_batched(marker, mask, offsets, erosion, dtype):
    for l in prange(marker.shape[0]):
        _propagate_3d(marker[l], mask[l], offsets, erosion, dtype)


def _propagate(marker, mask, offsets, erosion, working_dim, batch):
    dtype = marker.dtype
    if dtype == np.float16:
        NotImplementedError("numba-morph doesn't support float16 input! This is a limit of numba.")
    if batch:
        propagate_function = _propagate_2d_batched if working_dim == 2 else _propagate_3d_batched
    else:
        propagate_function = _propagate_2d if working_dim == 2 else _propagate_3d

    propagate_function(marker, mask, offsets, erosion, dtype.type)