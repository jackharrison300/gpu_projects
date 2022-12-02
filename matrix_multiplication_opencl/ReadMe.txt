Working environment: I used Palmetto cluster CPU with one NVIDIA V100 GPU

To compile: `./run_new matMultiply`
To run: `./matMultiply`

I chose not to print results as they were quite verbose, but they can be printed out by uncommenting the pertinent section (lines 164-172).

As shown in the chart, performance increases as BLOCK_SIZE increases up to size 4, then plateaus at size 8. This seems roughly to be expected, as we
would expect performance increase as we use more of warp (for FLOPS) and half-warp (for memory access), then plateaus once we are using all of warp.
However, there is a slight uptick in execution time between size 4 and size 8, which I do not know how to account for. I could have made performance increase
more linearly with BLOCK_SIZE by using shared memory, but that was not listed in the requirements for this assignment.