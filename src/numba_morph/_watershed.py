import numpy as np
from numba import jit, njit, prange
from heapq import heappush, heappop
from numba.typed import List

def _heapify_markers_2d(markers, image):
    coords = np.argwhere(markers != 0).astype(np.uint32)
    ncoords = coords.shape[0]
    if ncoords > 0:
        pixels = image[markers != 0]
        age = np.arange(ncoords, dtype=np.uint32)
        pq = [(pixels[i], age[i], coords[i, 0], coords[i, 1])
              for i in range(ncoords)]
        ordering = np.lexsort((age, pixels))
        pq = [pq[i] for i in ordering]
    else:
        pq = np.zeros((0, markers.ndim + 3), int)
    return (pq, ncoords)

def _heapify_markers_3d(markers, image):
    coords = np.argwhere(markers != 0).astype(np.uint32)
    ncoords = coords.shape[0]
    if ncoords > 0:
        pixels = image[markers != 0]
        age = np.arange(ncoords, dtype=np.uint32)
        pq = [(pixels[i], age[i], coords[i, 0], coords[i, 1], coords[i, 2])
              for i in range(ncoords)]
        ordering = np.lexsort((age, pixels))
        pq = [pq[i] for i in ordering]
    else:
        pq = np.zeros((0, markers.ndim + 3), int)
    return (pq, ncoords)


@njit(cache=True)
def _watershed_loop_2d(pq, labels, offsets, mask, image, age):
    H, W = labels.shape
    n_offs = offsets.shape[0] if offsets is not None else 0
    while len(pq):
        pix_value, pix_age, x, y = heappop(pq)   # age not needed after pop
        label = labels[x, y]
        for idx in range(n_offs):
            dx = offsets[idx, 0]
            dy = offsets[idx, 1]
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= H or ny < 0 or ny >= W:
                continue
            if labels[nx, ny] != 0:
                continue
            if mask is not None and not mask[nx, ny]:
                continue
            labels[nx, ny] = label
            heappush(pq, (image[nx, ny], np.uint32(age), np.uint32(nx), np.uint32(ny)))
            age += 1
    return labels, age


def _watershed_2d_single(image, markers, mask, offsets):
    """2D watershed, modifies markers in-place."""
    pq, age = _heapify_markers_2d(markers, image)
    _watershed_loop_2d(pq, markers, offsets, mask, image, age)



@njit(cache=True)
def _watershed_loop_3d(pq, labels, offsets, mask, image, age):
    D, H, W = labels.shape
    n_offs = offsets.shape[0] if offsets is not None else 0
    while len(pq) > 0:
        pix_value, pix_age, d, h, w = heappop(pq)
        label = labels[d, h, w]
        for idx in range(n_offs):
            dd = offsets[idx, 0]
            dh = offsets[idx, 1]
            dw = offsets[idx, 2]
            nd = d + dd
            nh = h + dh
            nw = w + dw
            if nd < 0 or nd >= D or nh < 0 or nh >= H or nw < 0 or nw >= W:
                continue
            if labels[nd, nh, nw] != 0:
                continue
            if mask is not None and not mask[nd, nh, nw]:
                continue
            labels[nd, nh, nw] = label
            heappush(pq, (image[nd, nh, nw], np.uint32(age), np.uint32(nd), np.uint32(nh), np.uint32(nw)))
            age += 1
    return labels, age


def _watershed_3d_single(image, markers, mask, offsets):
    pq, age = _heapify_markers_3d(markers, image)
    _watershed_loop_3d(pq, markers, offsets, mask, image, age)


# ----------------------------------------------------------------------
# Batched versions (parallel over first dimension)
# ----------------------------------------------------------------------

def build_all_pqs_2d(markers, image):
    """Build all initial heaps for a 2D batch.

    Returns:
        all_pqs: numba.typed.List of heaps (each heap is a list of tuples)
        all_ages: numba.typed.List of starting ages (one per batch slice)
    """
    L = markers.shape[0]
    all_pqs = List()
    all_ages = List()
    for l in range(L):
        pq, age = _heapify_markers_2d(markers[l], image[l])  # pure Python
        all_pqs.append(pq)  # Numba will convert to a typed list of tuples
        all_ages.append(age)
    return all_pqs, all_ages


def build_all_pqs_3d(markers, image):
    L = markers.shape[0]
    all_pqs = List()
    all_ages = List()
    for l in range(L):
        pq, age = _heapify_markers_3d(markers[l], image[l])
        all_pqs.append(pq)
        all_ages.append(age)
    return all_pqs, all_ages

@njit(parallel=True, cache=True)
def _watershed_2d_batch_parallel(image, markers, mask, offsets, all_pqs, all_ages):
    L = image.shape[0]
    for l in prange(L):
        pq = all_pqs[l]
        age = all_ages[l]
        img_slice = image[l]
        mk_slice = markers[l]
        mask_slice = mask[l] if mask is not None else None
        # The loop modifies mk_slice in‑place; we ignore the returned age
        _watershed_loop_2d(pq, mk_slice, offsets, mask_slice, img_slice, age)

@njit(parallel=True, cache=True)
def _watershed_3d_batch_parallel(image, markers, mask, offsets, all_pqs, all_ages):
    L = image.shape[0]
    for l in prange(L):
        pq = all_pqs[l]
        age = all_ages[l]
        img_slice = image[l]
        mk_slice = markers[l]
        mask_slice = mask[l] if mask is not None else None
        _watershed_loop_3d(pq, mk_slice, offsets, mask_slice, img_slice, age)


def _marker_controlled_watershed(image, markers, working_dim, mask=None, offsets=None, batch=False):
    if batch:
        if working_dim == 2:
            all_pqs, all_ages = build_all_pqs_2d(markers, image)
            _watershed_2d_batch_parallel(image, markers, mask, offsets, all_pqs, all_ages)
        elif working_dim == 3:
            all_pqs, all_ages = build_all_pqs_3d(markers, image)
            _watershed_3d_batch_parallel(image, markers, mask, offsets, all_pqs, all_ages)
    else:
        if working_dim == 2:
            _watershed_2d_single(image, markers, mask, offsets)
        elif working_dim == 3:
            _watershed_3d_single(image, markers, mask, offsets)
    return markers