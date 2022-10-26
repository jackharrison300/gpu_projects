#include <stdio.h>
#include <sys/time.h>

// to make this a bit more platform agnostic (though at a performance cost), we
// could make it an int set by querying device dynamically at run time. this would require
// us to use switch statements if using templated loop unrolling
#define THREADS_PER_BLOCK 1024

// Prints the specified message and quits
void die(const char *message) {
	printf("%s\n", message);
	exit(1);
}

// Returns the vector dot product of A and B
float CPU_big_dot(float *A, float *B, int N) {	
	// Compute the dot product
        float sum = 0;
	for (int i = 0; i < N; i++) sum += A[i] * B[i];
	
	// Return the result
	return sum;
}

// A GPU kernel that computes the vector dot product of A and B, down to the result per block
__global__ void kernel1(float *A, float *B, float *C, int N, float* g_odata) {
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// If we use array length of 2^n and 1024 threads per block, this is only necessary for N < 1024
	if (idx >= N) return;
	// Compute a single element of the result vector (if the element is valid)
	C[idx] = A[idx] * B[idx];

	__shared__ float smem[THREADS_PER_BLOCK];
	float *idata = C + blockIdx.x * blockDim.x;

	smem[tid] = idata[tid];
	__syncthreads();

	// these THREADS_PER_BLOCK checks aren't strictly necessary, but they help us avoid errors if we change THREADS_PER_BLOCK
	if (THREADS_PER_BLOCK >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();

	if (THREADS_PER_BLOCK >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();

	if (THREADS_PER_BLOCK >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();

	if (THREADS_PER_BLOCK >= 128 && tid < 64) smem[tid] += smem[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile float *vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

float dotproduct_1(float *A_CPU, float *B_CPU, int N) {
	// Determine the number of thread blocks in the grid 
	int blocks_per_grid = (int) ((float) (N + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);

	// Allocate GPU memory for the inputs and the result
	int vector_size = N * sizeof(float);
	int odata_size = blocks_per_grid * sizeof(float);
	float *A_GPU, *B_GPU, *C_GPU, *odata_GPU;

	if (cudaMalloc((void **) &A_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &B_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &C_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &odata_GPU, odata_size) != cudaSuccess) die("Error allocating GPU memory");

	// Transfer the input vectors to GPU memory
	cudaMemcpy(A_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, vector_size, cudaMemcpyHostToDevice);

	kernel1<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A_GPU, B_GPU, C_GPU, N, odata_GPU);

	// Check for kernel errors
	cudaError_t error = cudaGetLastError();
	if (error) {
		char message[256];
		sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
		die(message);
	}

	// Allocate CPU memory for the result
	float *odata_CPU = (float *) malloc(vector_size);
	if (odata_CPU == NULL) die("Error allocating CPU memory");

	// Transfer the result from the GPU to the CPU
	cudaMemcpy(odata_CPU, odata_GPU, odata_size, cudaMemcpyDeviceToHost);

	// Free the GPU memory
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
	cudaFree(odata_GPU);

	// Do the summation of multiplication in CPU
	float sum = 0;
	for (int i = 0; i < blocks_per_grid; i++) sum += odata_CPU[i];

	free(odata_CPU);
	
	return sum;
}

// A GPU kernel that computes the vector dot product of A and B
// (each thread computes the result per block)
__global__ void kernel2(float *A, float *B, float *C, int N, float *result) {
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// If we use array length of 2^n and 1024 threads per block, this is only necessary for N < 1024
	if (idx >= N) return;
	// Compute a single element of the result vector (if the element is valid)
	C[idx] = A[idx] * B[idx];

	__shared__ float smem[THREADS_PER_BLOCK];
	float *idata = C + blockIdx.x * blockDim.x;

	smem[tid] = idata[tid];
	__syncthreads();

	// these THREADS_PER_BLOCK checks aren't strictly necessary, but they help us avoid errors if we change THREADS_PER_BLOCK
	if (THREADS_PER_BLOCK >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();

	if (THREADS_PER_BLOCK >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();

	if (THREADS_PER_BLOCK >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();

	if (THREADS_PER_BLOCK >= 128 && tid < 64) smem[tid] += smem[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile float *vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) atomicAdd(result, smem[0]);
}

float dotproduct_2(float *A_CPU, float *B_CPU, int N) {
	// Determine the number of thread blocks in the grid 
	int blocks_per_grid = (int) ((float) (N + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);

	// Allocate GPU memory for the inputs and the result
	int vector_size = N * sizeof(float);
	int sum_start = 0;
	float *A_GPU, *B_GPU, *C_GPU, *result_GPU;

	if (cudaMalloc((void **) &A_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &B_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &C_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &result_GPU, sizeof(float)) != cudaSuccess) die("Error allocating GPU memory");

	// Transfer the input vectors to GPU memory
	cudaMemcpy(A_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(result_GPU, &sum_start, sizeof(float), cudaMemcpyHostToDevice);

	kernel2<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A_GPU, B_GPU, C_GPU, N, result_GPU);

	// Check for kernel errors
	cudaError_t error = cudaGetLastError();
	if (error) {
		char message[256];
		sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
		die(message);
	}

	float result_CPU;

	// Transfer the result from the GPU to the CPU
	cudaMemcpy(&result_CPU, result_GPU, sizeof(float), cudaMemcpyDeviceToHost);

	// Free the GPU memory
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
	cudaFree(result_GPU);

	return result_CPU;
}

// Returns a randomized vector containing N elements
float *get_random_vector(int N) {
	if (N < 1) die("Number of elements must be greater than zero");
	
	// Allocate memory for the vector
	float *V = (float *) malloc(N * sizeof(float));
	if (V == NULL) die("Error allocating CPU memory");
	
	// Populate the vector with random numbers
	for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();
	
	// Return the randomized vector
	return V;
}

int main(int argc, char **argv) {
	// Seed the random generator (use a constant here for repeatable results)
	srand(10);

	// Determine the vector length
	int N = 1 << 24;  // default value
	if (argc > 1) N = atoi(argv[1]); // user-specified value

	// Generate two random vectors
	float *A = get_random_vector(N);
	float *B = get_random_vector(N);
	
	cudaEvent_t start_1, stop_1, start_2, stop_2;
	cudaEventCreate(&start_1);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&start_2);
	cudaEventCreate(&stop_2);

	// Compute dot product using kernel1()
	cudaEventRecord(start_1);
	float kernel1_result = dotproduct_1(A, B, N);
	cudaEventRecord(stop_1);

	// Compute dot product using kernel2()
	cudaEventRecord(start_2);
	float kernel2_result = dotproduct_2(A, B, N);
	cudaEventRecord(stop_2);

	cudaDeviceSynchronize();
	// Compare execution time of kernel1 vs kernel2
	float kernel1_time_ms, kernel2_time_ms = 0;
    cudaEventElapsedTime(&kernel1_time_ms, start_1, stop_1);
	cudaEventElapsedTime(&kernel2_time_ms, start_2, stop_2);

	printf("\nkernel1 result: %.2f\n", kernel1_result);
	printf("\nkernel1 execution time: %.2f ms\n", kernel1_time_ms);
	
	printf("\nkernel2 result: %.2f\n", kernel2_result);
	printf("\nkernel2 execution time: %.2f ms\n", kernel2_time_ms);

	float cpu_result = CPU_big_dot(A, B, N);
	printf("\ncpu result: %.2f\n", cpu_result);

	if (kernel2_time_ms > kernel1_time_ms) printf("\nkernel1 outperformed kernel2 by %.2fx\n", (float) kernel2_time_ms / (float) kernel1_time_ms);
	else                     printf("\nkernel2 outperformed kernel1 by %.2fx\n", (float) kernel1_time_ms / (float) kernel2_time_ms);

	free(A);
	free(B);
}