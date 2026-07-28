import numpy as np
from numba import njit

@njit
def _voronoi_1d(site_pos, site_R, L):
    # Collect valid sites (site_pos != -1)
    # We'll store their indices in arrays f and g (stack)
    f = np.empty(L, dtype=np.int32)   # site index (position in array)
    n = 0
    for i in range(L):
        if site_pos[i] != -1:
            f[n] = i
            n += 1

    best_idx = np.full(L, -1, dtype=np.int32)
    if n == 0:
        return best_idx

    # Build the lower envelope using a stack
    stack = np.empty(n, dtype=np.int32)
    top = -1
    for i in range(n):
        # While stack has at least 2 sites, check if the last site is obsolete
        while top >= 1:
            # indices of the three sites: prev, last, new
            idx1 = f[stack[top - 1]]
            idx2 = f[stack[top]]
            idx3 = f[i]

            s1 = site_pos[idx1]
            s2 = site_pos[idx2]
            s3 = site_pos[idx3]
            R1 = site_R[idx1]
            R2 = site_R[idx2]
            R3 = site_R[idx3]

            # Intersection x between site2 and site3: (x - s2)^2 + R2 = (x - s3)^2 + R3
            # => x23 = (R3 - R2 + s3^2 - s2^2) / (2*(s3 - s2))
            x23 = (R3 - R2 + s3 * s3 - s2 * s2) / (2.0 * (s3 - s2))
            # Intersection between site1 and site2
            x12 = (R2 - R1 + s2 * s2 - s1 * s1) / (2.0 * (s2 - s1))

            if x23 <= x12:
                # last site (idx2) is never the best, pop it
                top -= 1
            else:
                break
        # Push new site
        top += 1
        stack[top] = i

    # Now the stack contains the sites forming the lower envelope.
    # Traverse the line and assign the best site.
    l = 0  # pointer into stack
    for x in range(L):
        # Move l forward while the next site gives a smaller distance
        while l < top:
            idx_cur = f[stack[l]]
            idx_next = f[stack[l + 1]]
            s_cur = site_pos[idx_cur]
            s_next = site_pos[idx_next]
            R_cur = site_R[idx_cur]
            R_next = site_R[idx_next]

            # distance to current site: (x - s_cur)^2 + R_cur
            # distance to next site: (x - s_next)^2 + R_next
            # If next is better (or equal), advance l
            if (x - s_next) ** 2 + R_next <= (x - s_cur) ** 2 + R_cur:
                l += 1
            else:
                break
        best_idx[x] = f[stack[l]]

    return best_idx


@njit
def edt_feature_transform_2d(binary):
    """
    Compute the exact Euclidean Feature Transform for a 2D binary image.

    Parameters
    ----------
    binary : 2D int array, shape (H, W)
        1 for foreground, 0 for background.

    Returns
    -------
    feat_r : 2D int array, shape (H, W)
        Row coordinate of the nearest background pixel for each pixel.
    feat_c : 2D int array, shape (H, W)
        Column coordinate of the nearest background pixel for each pixel.
        For background pixels, feat_r = row, feat_c = col (self).
    """
    H, W = binary.shape
    feat_r = np.full((H, W), -1, dtype=np.int32)
    feat_c = np.full((H, W), -1, dtype=np.int32)

    # Initialise: background pixels point to themselves, foreground to -1
    for i in range(H):
        for j in range(W):
            if binary[i, j] == 0:
                feat_r[i, j] = i
                feat_c[i, j] = j

    # ----- First pass: along rows (axis 0) -----
    # For each column, run 1D Voronoi on the rows.
    # At this stage, accumulated R = 0.
    for j in range(W):
        site_pos = feat_r[:, j]      # row coordinate of site (or -1)
        site_R = np.zeros(H, dtype=np.float64)  # no previous dims
        best_idx = _voronoi_1d(site_pos, site_R, H)
        for i in range(H):
            if best_idx[i] != -1:
                # best_idx[i] is the row index of the best site in this column
                # The site's coordinates are (feat_r[best_idx[i], j], j)
                feat_r[i, j] = feat_r[best_idx[i], j]
                feat_c[i, j] = j   # column is unchanged in this pass

    # ----- Second pass: along columns (axis 1) -----
    # For each row, run 1D Voronoi on the columns.
    # Now the accumulated R for each site is (row - feat_r[row, col])^2.
    for i in range(H):
        site_pos = feat_c[i, :]      # column coordinate of site (or -1)
        # compute R = (i - feat_r[i, j])^2 for each site
        site_R = np.zeros(W, dtype=np.float64)
        for j in range(W):
            if feat_r[i, j] != -1:
                dr = i - feat_r[i, j]
                site_R[j] = dr * dr
        best_idx = _voronoi_1d(site_pos, site_R, W)
        for j in range(W):
            if best_idx[j] != -1:
                # best_idx[j] is the column index of the best site in this row
                # Use that site's feature coordinates
                best_col = best_idx[j]
                feat_r[i, j] = feat_r[i, best_col]
                feat_c[i, j] = feat_c[i, best_col]

    return feat_r, feat_c

if __name__ == '__main__':
    # Example binary image (0 = background, 1 = foreground)
    binary = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.int32)

    feat_r, feat_c = edt_feature_transform_2d(binary)

    # Compute Euclidean distances (squared) using the feature coordinates
    # For each pixel, distance² = (i - feat_r[i,j])² + (j - feat_c[i,j])²
    dist2 = (np.indices(binary.shape)[0] - feat_r) ** 2 + \
            (np.indices(binary.shape)[1] - feat_c) ** 2
    dist = np.sqrt(dist2)  # exact Euclidean distance

    print(dist)