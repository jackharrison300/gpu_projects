__kernel void matrixMulKernel(__global float *Md, __global float *Nd, __global float *Pd, unsigned width) {
  size_t Row = get_global_id(1);
  size_t Col = get_global_id(0);

  float Pvalue = 0;
  for (int k = 0; k < width; k++) {
    Pvalue += Md[Row*width+k] * Nd[k*width+Col];
  }
  Pd[Row*width+Col] = Pvalue;
}
