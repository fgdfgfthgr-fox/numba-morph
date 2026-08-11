## v0.1.0
Initial release

## v0.1.1
Support using floating-point data type when performing local minimum and maximum calculations.

## v0.2.0
Lower the memory requirement of reconstruction in some cases.

Disable the "fast" algorithm for distance transform for now (since there are issues with it).

Add a shortcut to the core erosion and dilation algorithms (and therefore all the downstream functions) when the pixels 
are already at their minimum or maximum value.

Add "output" argument to cdt so the function can optionally do the ops in-place.