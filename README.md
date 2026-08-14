# numba-morph

[![Tests](https://github.com/fgdfgfthgr-fox/numba-morph/actions/workflows/python-app.yml/badge.svg)](https://github.com/fgdfgfthgr-fox/numba-morph/actions/workflows/python-app.yml/badge.svg)

A set of Numba-optimised morphological operations.

For very large arrays, is faster and uses less memory than their scikit-image or SciPy counterpart. 
Supports batched operations on both 2D and 3D.


## Features

* Numba accelerated: run at native machine code speed!
* Multi-threading & Batching: Parallelism both within spatial context or across leading dimensions.
* 2D & 3D with arbitrary leading dimensions.
* Versatile dtypes: Ops support various integer and float formats natively. There is no internal higher precision dtype.
* Lower memory use: Avoid padding or creating any unneeded intermediate arrays.
* All public functions are fully documented.

## List of available functions

* dilation, erosion
* distance_transform_cdt (Chamfer Distance Transform)
* morphological_gradient
* h_minima, h_maxima
* morphological_laplace
* local_minima, local_maxima
* opening, closing
* reconstruction
* white_tophat, black_tophat
* watershed
* welford_mean_std_w_mask

## Installation
You can install `numba-morph` directly from pypi using pip:

```
pip install numba-morph
```

## Example
```
import numpy as np
from numba_morph import dilation

original = np.random.randint(0, 8, (32, 64), dtype=np.uint8)
footprint = ndimage.generate_binary_structure(2, 2)
result = dilation(original, footprint=footprint)
```

## Benchmark
All testing were done on an AMD Ryzen 7 7735HS CPU.
You can replicate the result using the benchmark.py.

Note: all the time given does not include the start-up time needed for Numba to compile. 
This takes from a fraction of a second to a few seconds, depending on the testing ops.

Erosion (uint16):

| Array Size                | Time taken for numba-morph | Time taken for SciPy | Memory use (numba-morph) | Memory use (SciPy) |
|---------------------------|----------------------------|----------------------|--------------------------|--------------------|
| 64x64x64                  | **0.0046s**                | 0.0051s              | 0.014GB                  | **0.001GB**        |
| 256x256x256               | **0.1992s**                | 1.8260s              | 0.11GB                   | **0.09GB**         |
| 1024x1024                 | **0.0094s**                | 0.0185s              | 0.018GB                  | **0.006GB**        |
| 4096x4096                 | **0.1200s**                | 0.5038s              | 0.11GB                   | **0.09GB**         |
| 8x4096x4096 (2D, batched) | **0.7861s**                | 4.1044s              | **0.76GB**               | 0.78GB             |

Reconstruction (int16):

| Array Size                | Time taken for numba-morph | Time taken for scikit-image | Memory use (numba-morph) | Memory use (scikit-image) |
|---------------------------|----------------------------|-----------------------------|--------------------------|---------------------------|
| 8x64x64x64 (3D, batched)  | **0.4057s**                | 0.9805s                     | 0.039GB                  | **0.038GB**               |
| 8x1024x1024 (2D, batched) | **0.4165s**                | 3.3852s                     | **0.075GB**              | 0.143GB                   |

Chamfer Distance Transform (in=uint8, out=int32):

Note numba-morph don't have time advantage here. It's optimised for memory use. 
Another advantage of numba-morph not showing here is able to specify output directly in other dtype. e.g. smaller uint16

| Array Size                  | Time taken for numba-morph | Time taken for scikit-image | Memory use (numba-morph) | Memory use (scikit-image) |
|-----------------------------|----------------------------|-----------------------------|--------------------------|---------------------------|
| 256x256x256                 | 0.7417s                    | **0.4400s**                 | **0.109GB**              | 0.281GB                   |
| 512x512x512                 | 5.9335s                    | **3.0149s**                 | **0.762**                | 2.250GB                   |
| 1024x1024                   | 0.0269s                    | **0.0223**                  | 0.021GB                  | **0.018GB**               |
| 4096x4096                   | 0.3298s                    | **0.2715s**                 | **0.106GB**              | 0.281GB                   |
| 8x128x128x128 (3D, batched) | **0.2043s**                | 0.6633s                     | **0.113GB**              | 0.281GB                   |

## Limitations
* Does not support float16 inputs, this is due to [an issue with Numba](https://github.com/numba/numba/issues/4402).
* Speed gain is small or none for smaller inputs.
* Requires compiling, which means it's going to be slower if your images are small.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## AI use disclaimer

DeepSeek was used to generate and quality check the functions. 
All functions with AI involvement were manually thoroughly analysed, 
tested with data, modified and verified to satisfy desired input and output conditions. 
