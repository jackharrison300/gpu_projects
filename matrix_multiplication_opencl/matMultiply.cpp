#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

#define N 40
#define BLOCK_SIZE 1

char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}

int main(void) {
  cl_platform_id platform_id;
  cl_uint num_of_platforms = 0;
  cl_uint num_of_devices = 0;
  cl_device_id device_id;
  cl_context_properties properties[3];
  cl_int err;
  cl_context context;  
  cl_event prof_event;
  cl_command_queue command_queue;
  char *kernelSource;
  size_t kernelSize;
  cl_program program;
  cl_kernel kernel;
  cl_mem inputMatrix1_d, inputMatrix2_d, results_d;
  size_t global[2];
  size_t local[2];
  cl_double run_time;

  cl_float *inputMatrix1;
  cl_float *inputMatrix2;
  cl_float *results;
  cl_uint width = N;

  int x, y;
  int data = 0;

  inputMatrix1 = (cl_float *) malloc(sizeof(cl_float) * width * width);
  inputMatrix2 = (cl_float *) malloc(sizeof(cl_float) * width * width);
  results = (cl_float *) malloc(sizeof(cl_float) * width * width);

  for(y = 0; y < width; y++) {
    for(x = 0; x < width; x++) {
      inputMatrix1[y * width + x] = data;
      inputMatrix2[y * width + x] = data;
      results[y * width + x] = 0;
      data++;
    }
  }

  // Retrives a list of platforms available
  if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
    printf("Unable to get platform_id\n");
    return 1;
  }

  // Get a supported GPU device
  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, 
     &num_of_devices) != CL_SUCCESS) {
     printf("Unable to get device_id\n");
     return 1;
  }

  // Context properties list (must be terminated with 0)
  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties) platform_id;
  properties[2] = 0;

  // Create a context with the GPU device
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

  // Create a command queue using the context and device
  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

  // Load kernel file, prepend static info, and return total kernel size
  kernelSource = loadProgSource("matMultiply.cl", "", &kernelSize);

  // Create a program from the kernel source code
  program = clCreateProgramWithSource(context, 1, (const char **) 
            &kernelSource, NULL, &err);

  // Compile the program
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
     printf("Error building program\n");

     char buffer[4096];
     size_t length;
     clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
     printf("%s\n", buffer);
     exit(1);
  }

  // Specify which kernel from the program to execute
  kernel = clCreateKernel(program, "matrixMulKernel", &err);

  // Create buffers for the input and output
  inputMatrix1_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
          sizeof(float) * width * width, NULL, NULL);
  inputMatrix2_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
          sizeof(float) * width * width, NULL, NULL);
  results_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
          sizeof(float) * width * width, NULL, NULL);

  // Load data into the input buffers
  clEnqueueWriteBuffer(command_queue, inputMatrix1_d, CL_TRUE, 0,
                       sizeof(float) * width * width, inputMatrix1, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, inputMatrix2_d, CL_TRUE, 0,
                       sizeof(float) * width * width, inputMatrix2, 0, NULL, NULL);

  // Set the argument list for the kernel command
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputMatrix1_d);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputMatrix2_d);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &results_d);
  clSetKernelArg(kernel, 3, sizeof(cl_uint), &width);

  global[0] = width;
  global[1] = width;

  local[0] = BLOCK_SIZE;
  local[1] = BLOCK_SIZE;

  // Enqueue the kernel command for execution
  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global,
                         local, 0, NULL, &prof_event);
  clFinish(command_queue);

  err = clWaitForEvents(1, &prof_event);

  cl_ulong start_time, end_time;
  size_t return_bytes;

  err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
  err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);

  run_time = (double) (end_time - start_time);

  // divide run time by 1-^6 to convert from ns to ms
  printf("\nRun time of GPU matrix multiplication with size %d x %d (ms): %.6f\n", width, width, run_time / 1000000);

  // Copy the results from out of the output buffer
  clEnqueueReadBuffer(command_queue, results_d, CL_TRUE, 0,
                      sizeof(float) * width * width, results, 0, NULL, NULL);

  // Print the results - Commented out as it's fairly verbose at large scale, but could be used for verification at small scale
  // printf("\nGPU matrix multiplication result: \n");
  // for (int i = 0; i < width; i++) {
  //   for (int j = 0; j < width; j++) {
  //     printf("%f ", results[i * width + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // Cleanup (release OpenCL resources)
  clReleaseContext(context);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(inputMatrix1_d);
  clReleaseMemObject(inputMatrix2_d);
  clReleaseMemObject(results_d);

  return 0;
}
