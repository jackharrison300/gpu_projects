#include <cstddef>
#include <stdio.h>
#include <sys/time.h>

#define DIM 512

// next 2 functions copied from common.h to avoid dependency
#define CHECK(call)                                                            \
{                                                                              \
  const cudaError_t error = call;                                              \
  if (error != cudaSuccess)                                                    \
  {                                                                            \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
  }                                                                            \
}

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Forward function declarations
float CPU_big_dot(float *A, float *B, int N);
float *get_random_vector(int N);
void die(const char *message);

template <unsigned int iBlockSize>
__global__ void iKernel(float *A, float *B, float *C, unsigned int  N) {
  __shared__ float smem[DIM];

  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  float *idata = A + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < N)
  {
      float a1 = A[idx] * B[idx];
      float a2 = A[idx + blockDim.x] * B[idx + blockDim.x];
      float a3 = A[idx + 2 * blockDim.x] * B[idx + 2 * blockDim.x];
      float a4 = A[idx + 3 * blockDim.x] * B[idx + 3 * blockDim.x];
      float b1 = A[idx + 4 * blockDim.x] * B[idx + 4 * blockDim.x];
      float b2 = A[idx + 5 * blockDim.x] * B[idx + 5 * blockDim.x];
      float b3 = A[idx + 6 * blockDim.x] * B[idx + 6 * blockDim.x];
      float b4 = A[idx + 7 * blockDim.x] * B[idx + 7 * blockDim.x];
      A[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  smem[tid] = idata[tid];
  __syncthreads();

  // in-place reduction and complete unroll
  if (iBlockSize >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
  __syncthreads();

  if (iBlockSize >= 512 && tid < 256) smem[tid] += smem[tid + 256];
  __syncthreads();

  if (iBlockSize >= 256 && tid < 128) smem[tid] += smem[tid + 128];
  __syncthreads();

  if (iBlockSize >= 128 && tid < 64) smem[tid] += smem[tid + 64];
  __syncthreads();

  // unrolling warp
  if (tid < 32)
  {
      volatile float *vsmem = smem;
      vsmem[tid] += vsmem[tid + 32];
      vsmem[tid] += vsmem[tid + 16];
      vsmem[tid] += vsmem[tid +  8];
      vsmem[tid] += vsmem[tid +  4];
      vsmem[tid] += vsmem[tid +  2];
      vsmem[tid] += vsmem[tid +  1];
  }

  // write result for this block to global mem
  if (tid == 0) C[blockIdx.x] = smem[0];
}

int main(int argc, char **argv) {
  const int ngpus = 2;
  int ngpus_available;

  CHECK(cudaGetDeviceCount(&ngpus_available));
  printf("CUDA-capable devices: %d\n\n", ngpus_available);

  if (ngpus_available < ngpus) {
      fprintf(stderr, "Less than %d GPUs on this platform... exiting now", ngpus);
      exit(1);
  }

	// Seed the random generator (use a constant here for repeatable results)
	srand(10);

	// Determine the vector length
	const int N = 1 << 24;  // default value

	// Generate two random vectors
  float *h_A = get_random_vector(N);
  float *h_B = get_random_vector(N);
	
  double cpu_start = seconds();
	// Compute their dot product on the CPU
	float sum_CPU = CPU_big_dot(h_A, h_B, N);
  double cpu_total_time = seconds() - cpu_start;

  // record start time
  double gpu_start = seconds();

  // execution configuration
  const int blocksize = DIM;   // initial block size

  // Note this assumes the constraint that N is divisible by ngpus
  const int N_per_gpu  = N / ngpus;
  const size_t N_per_gpu_bytes = N_per_gpu * sizeof(float);

  dim3 block (blocksize, 1);
  dim3 grid  ((N_per_gpu + block.x - 1) / block.x, 1);
  const size_t num_blocksums_bytes = (grid.x / 8) * sizeof(float);

  // allocate device memory for the outer arrays of size ngpus (not the inner data arrays)
  float **d_A = (float **)malloc(sizeof(float *) * ngpus);
  float **d_B = (float **)malloc(sizeof(float *) * ngpus);  
  float **d_blocksums = (float **)malloc(sizeof(float *) * ngpus);
  float **h_blocksums = (float **)malloc(sizeof(float *) * ngpus);
  cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ngpus);

  for (int i = 0; i < ngpus; i++) {
    // set current device
    CHECK(cudaSetDevice(i));

    // allocate device memory
    CHECK(cudaMalloc((void **) &d_A[i], N_per_gpu_bytes));
    CHECK(cudaMalloc((void **) &d_B[i], N_per_gpu_bytes));
    CHECK(cudaMalloc((void **) &d_blocksums[i], num_blocksums_bytes));

    // allocate page locked host memory for asynchronous data transfer
    CHECK(cudaMallocHost((void **) &h_blocksums[i], num_blocksums_bytes));

    // create streams for timing and synchronizing
    CHECK(cudaStreamCreate(&stream[i]));
  }

  // distributing the workload across multiple devices
  for (int i = 0; i < ngpus; i++) {
    CHECK(cudaSetDevice(i));

    CHECK(cudaMemcpyAsync(d_A[i], (h_A + i * N_per_gpu), N_per_gpu_bytes, cudaMemcpyHostToDevice,
                          stream[i]));
    CHECK(cudaMemcpyAsync(d_B[i], (h_B + i * N_per_gpu), N_per_gpu_bytes, cudaMemcpyHostToDevice,
                          stream[i]));

    // Execute the kernel to compute the vector dot product on the GPU
    switch (blocksize) {
      case 1024:
        iKernel<1024><<<grid.x/8, block>>>(d_A[i], d_B[i], d_blocksums[i], N);
        break;
      case 512:
        iKernel<512><<<grid.x/8, block>>>(d_A[i], d_B[i], d_blocksums[i], N);
        break;
      case 256:
        iKernel<256><<<grid.x/8, block>>>(d_A[i], d_B[i], d_blocksums[i], N);
        break;
      case 128:
        iKernel<128><<<grid.x/8, block>>>(d_A[i], d_B[i], d_blocksums[i], N);
        break;
      case 64:
        iKernel<64><<<grid.x/8, block>>>(d_A[i], d_B[i], d_blocksums[i], N);
        break;
    }

    CHECK(cudaMemcpyAsync(h_blocksums[i], d_blocksums[i], num_blocksums_bytes, cudaMemcpyDeviceToHost,
                          stream[i]));
  }

  // synchronize streams
  for (int i = 0; i < ngpus; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaStreamSynchronize(stream[i]));
  }
  
  // Do the summation of block sums on CPU
  float sum_GPU = 0;
  for (int i = 0; i < ngpus; i++) {
	  for (int j = 0; j < grid.x / 8; j++) {
      sum_GPU += h_blocksums[i][j];
    }
  }
  
  // calculate the elapsed time in seconds
  double gpu_total_time = seconds() - gpu_start;

  // Cleanup
  CHECK(cudaFreeHost(h_A));
  CHECK(cudaFreeHost(h_B));
  for (int i = 0; i < ngpus; i++) {
      CHECK(cudaSetDevice(i));

      CHECK(cudaFree(d_A[i]));
      CHECK(cudaFree(d_B[i]));
      CHECK(cudaFree(d_blocksums[i]));

      CHECK(cudaFreeHost(h_blocksums[i]));

      CHECK(cudaStreamDestroy(stream[i]));

      CHECK(cudaDeviceReset());
  }

  free(d_A);
  free(d_B);
  free(d_blocksums);
  free(h_blocksums);
  free(stream);

  // print results
  printf("dot product with CPU = %f\n", sum_CPU);
  printf("dot product with GPU = %f\n", sum_GPU);


  printf("CPU total time:    %.2f ms \n", cpu_total_time * 1000.0);
  printf("%d GPUs total time: %.2f ms \n", ngpus, gpu_total_time * 1000.0);
  printf("GPU speedup: %.2f \n\n", cpu_total_time / gpu_total_time);

  printf("total array size %d M, using %d devices with each device "
          "handling %d M\n", N / 1024 / 1024, ngpus, N_per_gpu / 1024 / 1024);
  printf("grid %d block %d\n", grid.x, block.x);

  return EXIT_SUCCESS;
}

// Returns the vector dot product of A and B
float CPU_big_dot(float *A, float *B, int N) {	
	// Compute the dot product
        float sum = 0;
	for (int i = 0; i < N; i++) sum += A[i] * B[i];
	
	// Return the result
	return sum;
}

// Returns a randomized vector containing N elements
float *get_random_vector(int N) {
	if (N < 1) die("Number of elements must be greater than zero");

	// Allocate memory for the vector
	float *V;
  // Pin memory to enable async memcpy
  CHECK(cudaMallocHost((void **) &V, N * sizeof(float)));
	if (V == NULL) die("Error allocating CPU memory");
	
	// Populate the vector with random numbers
	for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();

	// Return the randomized vector
	return V;
}

// Prints the specified message and quits
void die(const char *message) {
	printf("%s\n", message);
	exit(1);
}
