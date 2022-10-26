#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "book.h"

#define NUM_ELEMENTS 1024*1024

// Returns a randomized vector containing N elements
float *get_random_vector(int N) {
  if (N < 1) fprintf(stderr, "Number of elements must be greater than zero");
  // Allocate memory for the vector
  float *V = (float *) malloc(N * sizeof(float));
  if (V == NULL) fprintf(stderr, "Error allocating CPU memory");
  // Populate the vector with random Numbers
  for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();
  // Return the randomized vector
  return V;
}

long long start_timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec; 
}

long long stop_timer(long long start_time, char *name) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
  printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / 
    (1000 * 1000));
  return end_time - start_time;
}

__global__ void multiply_gpu(float *a, float *b, float *c, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N)
    c[index] = a[index] * b[index];
}

void multiply_cpu(float *a, float *b, float *c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

float sum_array_cpu(float *x, int size) {
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += x[i];
  }
  return sum;
}

float CPU_big_dot(float *A, float *B, int N) {
  float *C = (float *) malloc(N * sizeof(float));
  multiply_cpu(A, B, C, N);
  float dot_product = sum_array_cpu(C, N);
  free(C);
  return dot_product;
}

float GPU_big_dot(float *A, float *B, int N) {
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
  const int threadsPerBlock = prop.maxThreadsPerBlock;

  float *d_a, *d_b, *d_c; // device copies of a, b, c
  const int size = N * sizeof(float);

  long long alloc_transfertogpu_start = start_timer();

  // Allocate space for device copies of a, b, c
  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);
  cudaMalloc((void**) &d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

  stop_timer(alloc_transfertogpu_start, (char *) "Memory allocation and data transfer from CPU to GPU time");

  long long kernel_start = start_timer();
  // Launch multiply_gpu() kernel on GPU with N threads 
  multiply_gpu<<<(N + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
  stop_timer(kernel_start, (char *) "Kernel execution time");

  // Copy result back to host
  float *C = (float *) malloc(N * sizeof(float));
  long long transfertocpu_start = start_timer();
  cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);
  stop_timer(transfertocpu_start, (char *) "Data transfer from GPU to CPU time");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  float dot_product = sum_array_cpu(C, N);

  free(C);

  return dot_product;
}

// a gpu function that can be iterated to sum an array in parallel
// prerequisite: array c has size >= 2*half_n,
__global__ void add_by_other_half_gpu(float *c, int half_n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < half_n)
    c[index] += c[index + half_n];
}

// an alternative implementation of GPU big dot that uses GPU parallelism
// in summing the array as well as in the multiplication step
float GPU_big_dot_alt(float *A, float *B, int N) {
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
  const int threadsPerBlock = prop.maxThreadsPerBlock;

  float *d_a, *d_b, *d_c; // device copies of a, b, c
  const int size = N * sizeof(float);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);
  cudaMalloc((void**) &d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

  // Launch multiply_gpu() kernel on GPU with N threads
  multiply_gpu<<<(N + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  int *indxs_to_add = (int *) malloc(((size_t) (log2(N) + 1)) * sizeof(int));
  int indxs_to_add_counter = 0;
  if (N % 2) {
    indxs_to_add[0] = N - 1;
    indxs_to_add_counter++;
  }
  for (int i = N/2; i >= 1; i /= 2) {
    add_by_other_half_gpu<<<(i + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_c, i);
    cudaDeviceSynchronize();
    if (i % 2) {
      indxs_to_add[indxs_to_add_counter] = i - 1;
      indxs_to_add_counter++;
    }
  }

  // Copy result back to host
  float *C = (float *) malloc(N * sizeof(float));
  cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  float dot_product = 0.0f;
  for (int i = 0; i < indxs_to_add_counter; i++) {
    dot_product += C[indxs_to_add[i]];
  }

  free(indxs_to_add);
  free(C);

  return dot_product;
}

int main(void) {
  float *a, *b; // host copies of a, b
  a = get_random_vector(NUM_ELEMENTS);
  b = get_random_vector(NUM_ELEMENTS);

  long long cpu_start = start_timer();
  float cpu_dot_prod = CPU_big_dot(a, b, NUM_ELEMENTS);
  long long cpu_time = stop_timer(cpu_start, (char *) "CPU total time");

  long long gpu_start = start_timer();
  float gpu_dot_prod = GPU_big_dot(a, b, NUM_ELEMENTS);
  // Note that we should not test both GPU_big_dot implementations in the same run,
  //   as the second pass of memory transfer will have less overhead and make the
  //   performance of the second pass appear artificially better. In other words,
  //   if we uncomment the alt GPU implementation code, we should comment out the
  //   first GPU implementation code
  // float gpu_dot_prod_alt = GPU_big_dot_alt(a, b, NUM_ELEMENTS);
  long long gpu_time = stop_timer(gpu_start, (char *) "GPU total time");

  free(a);
  free(b);

  printf("Speedup: %f\n", (float) cpu_time / (float) gpu_time);
  printf("CPU dot product: %f\n", cpu_dot_prod);
  printf("GPU dot product: %f\n", gpu_dot_prod);
  // Note that the alt implementation uses a different order of addition, and so
  //   the result will be a bit different (in fact, a bit more accurate)
  //   due to the order-sensitivity of floating-point arithmetic
  // printf("GPU alt dot product = %f\n", gpu_dot_prod_alt);
  if (cpu_dot_prod == gpu_dot_prod) {
    printf("CPU result equals GPU result\n");
  }
  else {
    printf("Error: CPU result does not equal GPU result\n");
  }

  return 0;
}
