# numba-morph

[![Tests](https://github.com/fgdfgfthgr-fox/numba-morph/actions/workflows/python-app.yml/badge.svg)](https://github.com/fgdfgfthgr-fox/numba-morph/actions/workflows/python-app.yml/badge.svg))

A set of Numba-optimised morphological operations.

Faster and uses less memory than their scikit-image or SciPy counterpart. Supports batched operations on both 2D and 3D.


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
* local_minima, local_maxima *
* opening, closing
* reconstruction
* white_tophat, black_tophat
* watershed
* welford_mean_std_w_mask

\* Under development, currently only works with integer dtype.

## Installation
~~You can install `numba-morph` directly from pypi using pip:~~
(Still working on it! Give me a few days!)

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

Dilation (int16):

| Array Size                | Time taken for numba-morph | Time taken for SciPy | Memory use (numba-morph) | Memory use (SciPy) |
|---------------------------|----------------------------|----------------------|--------------------------|--------------------|
| 64x64x64                  | **0.0072s**                | 0.0076s              | 0.022GB                  | **0.001GB**        |
| 128x128x128               | **0.0461s**                | 0.1335s              | 0.033GB                  | **0.012GB**        |
| 256x256x256               | **0.3823s**                | 2.0027s              | 0.12GB                   | **0.09GB**         |
| 512x512x512               | **2.5800s**                | 25.7317s             | 0.77GB                   | **0.75GB**         |
| 256x256                   | **0.0006s**                | 0.0014s              | 0.02GB                   | **~0.0GB**         |
| 1024x1024                 | **0.0101s**                | 0.0214s              | 0.026GB                  | **0.006GB**        |
| 4096x4096                 | **0.1617s**                | 0.5816s              | 0.11GB                   | **0.09GB**         |
| 8x4096x4096 (2D, batched) | **1.0951s**                | 4.9948s              | 0.77GB                   | **0.75GB**         |

Reconstruction (int16):

| Array Size                | Time taken for numba-morph | Time taken for scikit-image | Memory use (numba-morph) | Memory use (scikit-image) |
|---------------------------|----------------------------|-----------------------------|--------------------------|---------------------------|
| 64x64x64                  | 0.1343s                    | **0.1127s**                 | **0.024GB**              | 0.025GB                   |
| 128x128x128               | **1.3726s**                | 1.8128s                     | **0.040GB**              | 0.192GB                   |
| 256x256                   | **0.0046s**                | 0.0103s                     | **0.002GB**              | 0.006GB                   |
| 1024x1024                 | **0.0897s**                | 0.3546s                     | **0.029GB**              | 0.092GB                   |
| 8x64x64x128 (3D, batched) | **1.0468s**                | 2.2300s                     | **0.053GB**              | 0.075GB                   |
| 8x2048x2048 (2D, batched) | **2.0700s**                | 18.5615s                    | **0.271GB**              | 0.571GB                   |

Chamfer Distance Transform (in=uint8, out=int32):
Note numba-morph don't have time advantage here. It's optimised for memory use. 
Another advantage of numba-morph not showing here is able to specify output directly in other dtype. e.g. smaller uint16

| Array Size                  | Time taken for numba-morph | Time taken for scikit-image | Memory use (numba-morph) | Memory use (scikit-image) |
|-----------------------------|----------------------------|-----------------------------|--------------------------|---------------------------|
| 64x64x64                    | 0.0123s                    | **0.0059s**                 | 0.017GB                  | **0.004GB**               |
| 256x256x256                 | 0.4401s                    | **0.3687s**                 | **0.175GB**              | 0.266GB                   |
| 512x1024x1024               | 10.1897s                   | **5.9755s**                 | **2.519GB**              | 4.250GB                   |
| 512x512                     | 0.0058s                    | **0.0042s**                 | 0.017GB                  | **0.004GB**               |
| 4096x4096                   | **0.2333s**                | 0.2606s                     | **0.176GB**              | 0.266GB                   |
| 8x128x128x256 (3D, batched) | **0.5874s**                | 1.2909s                     | **0.328GB**              | 0.531GB                   |
| 8x4096x4096 (2D, batched)   | **1.5051s**                | 2.9589s                     | **1.269GB**              | 2.125GB                   |

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
