Assignment 2

To run code: nvcc dotProduct.cu

kernel1 outperforms kernel2 slightly. I suspect this is because, while
both versions do a serialized add of the block totals, there is more overhead
with the on-device serialized add from shared to global memory.