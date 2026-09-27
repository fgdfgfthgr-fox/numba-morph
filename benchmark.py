import numpy as np
import time
import timeit
import tracemalloc
#import numba_morph
from src import numba_morph
import skimage.morphology as morph
import scipy.ndimage as ndimage
import imageio.v3 as imageio
from skimage.data import binary_blobs
from scipy.ndimage import gaussian_filter


SHAPE = (256, 256, 256)
REPEATS = 2
OP_My = numba_morph.reconstruction
OP_Control = morph.reconstruction
footprint = ndimage.generate_binary_structure(3, 3)


def gaussian_random_field(sigma=4):
    # Larger sigma = less difficulty. At sigma=1 it's pure noise.
    noise = np.random.randn(*SHAPE).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=sigma)          # correlation length ≈ 8 voxels
    mask = np.clip((smooth - smooth.min()) /
                   (smooth.max() - smooth.min()) * 255, 0, 255).astype(np.uint16)
    marker = mask.copy()
    marker[1:-1, 1:-1] = marker.min()
    return mask, marker

def piecewise_constant_blobs():
    D, H, W = SHAPE
    mask = np.zeros((D, H, W), dtype=np.uint16)
    rng = np.random.default_rng(0)
    for _ in range(30):
        c = rng.integers(40, D-40, size=3)
        r = rng.integers(20, 60)
        val = rng.integers(100, 256)
        zz, yy, xx = np.indices((D, H, W))
        inside = ((zz-c[0])**2 + (yy-c[1])**2 + (xx-c[2])**2) < r*r
        mask[inside] = np.maximum(mask[inside], val)
    marker = mask.copy()
    marker[1:-1, 1:-1] = marker.min()
    return mask, marker


def op_my_run():
    mask, marker = gaussian_random_field()
    imageio.imwrite('mask.tiff', mask)
    result = OP_My(mask, marker, footprint=footprint)
    return result

def op_control_run():
    mask, marker = gaussian_random_field()
    result = OP_Control(marker, mask, footprint=footprint)
    return result

if __name__ == '__main__':
    tracemalloc.start()
    op_my_run()
    start_time = time.time()
    for i in range(REPEATS):
        result_1 = op_my_run()
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nnumba-morph:")
    print(f"  Time: {(end_time - start_time)/REPEATS:.4f} seconds")
    print(f"  Peak memory (tracemalloc): {peak / (1024 ** 3):.3f} GB")

    tracemalloc.start()
    op_control_run()
    start_time = time.time()
    for i in range(REPEATS):
        result_2 = op_control_run()
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nSciPy:")
    print(f"  Time: {(end_time - start_time)/REPEATS:.4f} seconds")
    print(f"  Peak memory (tracemalloc): {peak / (1024 ** 3):.3f} GB")