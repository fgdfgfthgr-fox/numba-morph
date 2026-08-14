import numpy as np
import time
import timeit
import tracemalloc
import numba_morph
import skimage.morphology as morph
import scipy.ndimage as ndimage

SHAPE = (4096, 4096)
REPEATS = 2
OP_My = numba_morph.distance_transform_cdt
OP_Control = ndimage.distance_transform_cdt
footprint = ndimage.generate_binary_structure(2, 2)

def op_my_run():
    input = np.random.randint(0, 2, SHAPE, dtype=np.uint16)
    #input_2 = np.random.randint(0, 256, SHAPE, dtype=np.uint16)
    #input_2 = np.minimum(input_2, input)
    #input_2 = input.copy()
    #input_2[1:-1, 1:-1] = input.min()
    result = OP_My(input, weights=(1,1))
    return result

def op_control_run():
    input = np.random.randint(0, 2, SHAPE, dtype=np.uint16)
    #input_2 = np.random.randint(0, 256, SHAPE, dtype=np.uint16)
    #input_2 = np.minimum(input_2, input)
    #input_2 = input.copy()
    #input_2[1:-1, 1:-1] = input.min()
    #result = np.zeros_like(input)
    #for i in range(result.shape[0]):
    #    result[i] = OP_Control(input[i])
    result = OP_Control(input, indices=(0,1))
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