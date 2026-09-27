import numpy as np
from numba import njit, prange

EULER_LUT = np.zeros(256, dtype=np.intc)
EULER_LUT[1::2] = np.array([
    1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    -7, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
    1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1
])

# 27 offsets (including centre at index 13)
OFFSETS_27 = np.array([
    (-1, -1, -1), (-1, 0, -1), (-1, 1, -1),
    (-1, -1, 0), (-1, 0, 0), (-1, 1, 0),
    (-1, -1, 1), (-1, 0, 1), (-1, 1, 1),
    (0, -1, -1), (0, 0, -1), (0, 1, -1),
    (0, -1, 0), (0, 0, 0), (0, 1, 0),
    (0, -1, 1), (0, 0, 1), (0, 1, 1),
    (1, -1, -1), (1, 0, -1), (1, 1, -1),
    (1, -1, 0), (1, 0, 0), (1, 1, 0),
    (1, -1, 1), (1, 0, 1), (1, 1, 1)
], dtype=np.int32)

# 26 offsets (exclude centre, index 13)
OFFSETS_26 = np.concatenate((OFFSETS_27[:13], OFFSETS_27[14:]))

# ---------- 2D constants ----------
OFFSETS_8 = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),  (0, 0),  (0, 1),
    (1, -1),  (1, 0),  (1, 1)
], dtype=np.int32)

OFFSETS_8_NO_CENTRE = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
], dtype=np.int32)


def _build_euler_lut_2d():
    """Build 2D Euler characteristic lookup table for 3x3 neighbourhood."""
    lut = np.zeros(512, dtype=np.intc)

    def bit(code, i):
        return (code >> i) & 1

    for code in range(512):
        # foreground 8-connected component count
        visited_fg = [False] * 9
        fg_comp = 0
        for i in range(9):
            if bit(code, i) and not visited_fg[i]:
                fg_comp += 1
                stack = [i]
                visited_fg[i] = True
                while stack:
                    cur = stack.pop()
                    cr, cc = cur // 3, cur % 3
                    for j in range(9):
                        if bit(code, j) and not visited_fg[j]:
                            jr, jc = j // 3, j % 3
                            if max(abs(cr - jr), abs(cc - jc)) <= 1:
                                visited_fg[j] = True
                                stack.append(j)

        # background 4-connected holes (components not touching border)
        visited_bg = [False] * 9
        holes = 0
        for i in range(9):
            if not bit(code, i) and not visited_bg[i]:
                stack = [i]
                visited_bg[i] = True
                touches_border = False
                while stack:
                    cur = stack.pop()
                    cr, cc = cur // 3, cur % 3
                    if cr == 0 or cr == 2 or cc == 0 or cc == 2:
                        touches_border = True
                    for j in range(9):
                        if not bit(code, j) and not visited_bg[j]:
                            jr, jc = j // 3, j % 3
                            if abs(cr - jr) + abs(cc - jc) == 1:
                                visited_bg[j] = True
                                stack.append(j)
                if not touches_border:
                    holes += 1

        lut[code] = fg_comp - holes

    return lut


EULER_LUT_2D = _build_euler_lut_2d()


# ---------- 2D helper functions (no allocations) ----------
@njit(cache=True, inline="always")
def _fill_neighbour_2d(img, r, c, neigh):
    """Fill a pre-allocated 9‑element array with the 3x3 neighbourhood."""
    for idx in range(9):
        dr, dc = OFFSETS_8[idx]
        rr, cc = r + dr, c + dc
        if 0 <= rr < img.shape[0] and 0 <= cc < img.shape[1]:
            neigh[idx] = img[rr, cc]
        else:
            neigh[idx] = 0


@njit(cache=True, inline="always")
def _is_endpoint_2d(neigh):
    return neigh.sum() == 2


@njit(cache=True, inline="always")
def _is_Euler_invariant_2d(neigh):
    code = 0
    for i in range(9):
        if neigh[i] == 1:
            code |= (1 << i)
    before = EULER_LUT_2D[code]
    after_code = code & ~(1 << 4)   # clear centre
    after = EULER_LUT_2D[after_code]
    return before == after


@njit(cache=True, inline="always")
def _is_simple_point_2d(neigh, visited):
    """visited is a pre-allocated array of size 8."""
    # Reset visited
    visited[:] = 0
    comp = 0

    for i in range(8):
        neigh_idx = i if i < 4 else i + 1
        if neigh[neigh_idx] == 1 and visited[i] == 0:
            comp += 1
            if comp > 1:
                return False

            stack = [i]
            visited[i] = 1
            while stack:
                cur = stack.pop()
                cur_r, cur_c = OFFSETS_8_NO_CENTRE[cur]
                for j in range(8):
                    if visited[j] == 0:
                        j_neigh_idx = j if j < 4 else j + 1
                        if neigh[j_neigh_idx] == 1:
                            j_r, j_c = OFFSETS_8_NO_CENTRE[j]
                            if abs(cur_r - j_r) <= 1 and abs(cur_c - j_c) <= 1:
                                visited[j] = 1
                                stack.append(j)
    return comp == 1


@njit
def _skeletonize_2d(img):
    shape = img.shape
    borders = [1, 2, 3, 4]   # N, S, E, W
    unchanged_borders = 0

    # Pre-allocate reusable arrays
    neigh = np.empty(9, dtype=np.uint8)
    visited = np.zeros(8, dtype=np.uint8)

    while unchanged_borders < 4:
        unchanged_borders = 0

        for border in borders:
            candidates = []

            for r in range(shape[0]):
                for c in range(shape[1]):
                    if img[r, c] != 1:
                        continue

                    on_border = False
                    if border == 1:      # N (c-1)
                        on_border = (c == 0 or img[r, c - 1] == 0)
                    elif border == 2:    # S (c+1)
                        on_border = (c == shape[1] - 1 or img[r, c + 1] == 0)
                    elif border == 3:    # E (r+1)
                        on_border = (r == shape[0] - 1 or img[r + 1, c] == 0)
                    elif border == 4:    # W (r-1)
                        on_border = (r == 0 or img[r - 1, c] == 0)

                    if not on_border:
                        continue

                    _fill_neighbour_2d(img, r, c, neigh)
                    if _is_endpoint_2d(neigh):
                        continue
                    if not _is_Euler_invariant_2d(neigh):
                        continue
                    if not _is_simple_point_2d(neigh, visited):
                        continue

                    candidates.append((r, c))

            no_change = True
            for r, c in candidates:
                _fill_neighbour_2d(img, r, c, neigh)
                if _is_simple_point_2d(neigh, visited):
                    img[r, c] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1


# ---------- 3D helper functions (no allocations) ----------
@njit(cache=True, inline="always")
def _fill_neighbour_3d(img, p, r, c, neigh):
    """Fill a pre-allocated 27‑element array with the 3x3x3 neighbourhood."""
    for idx in range(27):
        dp, dr, dc = OFFSETS_27[idx]
        pp, rr, cc = p + dp, r + dr, c + dc
        if 0 <= pp < img.shape[0] and 0 <= rr < img.shape[1] and 0 <= cc < img.shape[2]:
            neigh[idx] = img[pp, rr, cc]
        else:
            neigh[idx] = 0


@njit(cache=True, inline="always")
def _is_endpoint_3d(neigh):
    return neigh.sum() == 2


@njit(cache=True, inline="always")
def _is_Euler_invariant_3d(neigh):
    OCTANT_INDICES = (
        (2, 1, 11, 10, 5, 4, 14),
        (0, 9, 3, 12, 1, 10, 4),
        (8, 7, 17, 16, 5, 4, 14),
        (6, 15, 7, 16, 3, 12, 4),
        (20, 23, 19, 22, 11, 14, 10),
        (18, 21, 9, 12, 19, 22, 10),
        (26, 23, 17, 14, 25, 22, 16),
        (24, 25, 15, 16, 21, 22, 12),
    )
    euler = 0
    for octant in OCTANT_INDICES:
        n = 1
        for j, idx in enumerate(octant):
            if neigh[idx] == 1:
                n |= (1 << (7 - j))
        euler += EULER_LUT[n]
    return euler == 0


@njit(cache=True, inline="always")
def _is_simple_point_3d(neigh, cube, visited):
    """cube (26) and visited (26) are pre-allocated arrays."""
    # Fill cube from neigh, excluding centre (index 13)
    for i in range(13):
        cube[i] = neigh[i]
    for i in range(13, 26):
        cube[i] = neigh[i + 1]

    # Reset visited
    visited[:] = 0

    comp = 0
    for i in range(26):
        if cube[i] == 1 and visited[i] == 0:
            comp += 1
            if comp > 1:
                return False

            stack = [i]
            visited[i] = 1
            while stack:
                cur = stack.pop()
                cur_dp, cur_dr, cur_dc = OFFSETS_26[cur]
                for j in range(26):
                    if cube[j] == 1 and visited[j] == 0:
                        j_dp, j_dr, j_dc = OFFSETS_26[j]
                        if (abs(cur_dp - j_dp) <= 1 and
                            abs(cur_dr - j_dr) <= 1 and
                            abs(cur_dc - j_dc) <= 1):
                            visited[j] = 1
                            stack.append(j)
    return comp == 1


@njit(cache=True)
def _skeletonize_3d(img):
    shape = img.shape
    borders = [4, 3, 2, 1, 5, 6]
    unchanged_borders = 0

    # Pre-allocate reusable arrays
    neigh = np.empty(27, dtype=np.uint8)
    cube = np.empty(26, dtype=np.uint8)
    visited = np.zeros(26, dtype=np.uint8)

    while unchanged_borders < 6:
        unchanged_borders = 0

        for border in borders:
            candidates = []

            for p in range(shape[0]):
                for r in range(shape[1]):
                    for c in range(shape[2]):
                        if img[p, r, c] != 1:
                            continue

                        on_border = False
                        if border == 1:      # N (c-1)
                            on_border = (c == 0 or img[p, r, c - 1] == 0)
                        elif border == 2:    # S (c+1)
                            on_border = (c == shape[2] - 1 or img[p, r, c + 1] == 0)
                        elif border == 3:    # E (r+1)
                            on_border = (r == shape[1] - 1 or img[p, r + 1, c] == 0)
                        elif border == 4:    # W (r-1)
                            on_border = (r == 0 or img[p, r - 1, c] == 0)
                        elif border == 5:    # U (p+1)
                            on_border = (p == shape[0] - 1 or img[p + 1, r, c] == 0)
                        elif border == 6:    # B (p-1)
                            on_border = (p == 0 or img[p - 1, r, c] == 0)

                        if not on_border:
                            continue

                        _fill_neighbour_3d(img, p, r, c, neigh)
                        if _is_endpoint_3d(neigh):
                            continue
                        if not _is_Euler_invariant_3d(neigh):
                            continue
                        if not _is_simple_point_3d(neigh, cube, visited):
                            continue

                        candidates.append((p, r, c))

            no_change = True
            for p, r, c in candidates:
                _fill_neighbour_3d(img, p, r, c, neigh)
                if _is_simple_point_3d(neigh, cube, visited):
                    img[p, r, c] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1


# ---------- main internal loop ----------
@njit(parallel=True, cache=True)
def _skeletonize_internal_loop(img, dim):
    if dim == 2:
        if img.ndim == 2:
            _skeletonize_2d(img)
        elif img.ndim == 3:
            for idx in prange(img.shape[0]):
                _skeletonize_2d(img[idx])

    elif dim == 3:
        if img.ndim == 3:
            _skeletonize_3d(img)
        elif img.ndim == 4:
            for idx in prange(img.shape[0]):
                _skeletonize_3d(img[idx])