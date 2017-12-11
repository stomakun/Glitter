#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#include <vector>
#include <chrono>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array
//
const char *KernelSource = "\n" \
"__kernel void mult(                                                    \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int N)                                               \n" \
"{                                                                      \n" \
"   int idx = get_global_id(0);                                         \n" \
"   if (idx >= N * N) return;                                           \n" \
"   int row = idx / N;                                                  \n" \
"   int col = idx % N;                                                  \n" \
"   float v = 0.0;                                                      \n" \
"   for (unsigned int i = 0; i < N; i++) {                              \n" \
"     v += a[row * N + i] * b[i * N + col];                             \n" \
"   }                                                                   \n" \
"   c[idx] = v;                                                         \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

int opencl(GLfloat *a, GLfloat *b, GLfloat *c, int iters, unsigned int N, int gpu = 1)
{
    int err;                            // error code returned from api calls
    unsigned int count = N * N;

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem ma, mb;                       // device memory used for the input array
    cl_mem mc;                      // device memory used for the output array

    // Connect to a compute device
    //
    err = clGetDeviceIDs(nullptr, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context
    //
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, &KernelSource, nullptr, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "mult", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    ma = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, nullptr, nullptr);
    mb = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, nullptr, nullptr);
    mc = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(float) * count, nullptr, nullptr);
    if (!ma || !mb || !mc)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, ma, CL_TRUE, 0, sizeof(float) * count, a, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(commands, mb, CL_TRUE, 0, sizeof(float) * count, b, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    auto opencl_start = std::chrono::system_clock::now();
    for (int iter = 0; iter < iters; ++iter) {

        // Set the arguments to our compute kernel
        //
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ma);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mb);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mc);
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }

        // Get the maximum work group size for executing the kernel on the device
        //
        err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to retrieve kernel work group info! %d\n", err);
            exit(1);
        }

        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        global = count;
        err = clEnqueueNDRangeKernel(commands, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (err) {
            printf("Error: Failed to execute kernel!\n");
            return EXIT_FAILURE;
        }

        // Wait for the command commands to get serviced before reading back results
        //
        clFinish(commands);

        // Read back the results from the device to verify the output
        //
        err = clEnqueueReadBuffer(commands, mc, CL_TRUE, 0, sizeof(float) * count, c, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
    }
    auto opencl_end = std::chrono::system_clock::now();
    std::cout << "opencl (gpu = " << gpu << "): "
              << (std::chrono::duration_cast<std::chrono::microseconds>(opencl_end - opencl_start).count() / iters)
              << std::endl;

    // Shutdown and cleanup
    //
    clReleaseMemObject(ma);
    clReleaseMemObject(mb);
    clReleaseMemObject(mc);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

