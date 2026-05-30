#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

const int LENGTH = 1000000;

const char* source
    = "__kernel void saxpy(const __global float *X, \n"
      "                    __global float *Y,       \n"
      "                    const float a) {         \n"
      "    uint gid = get_global_id(0);             \n"
      "    Y[gid]   = a * X[gid] + Y[gid];          \n"
      "}                                            \n";

int main()
{
    cl_uint num_platforms;
    cl_uint num_devices;
    cl_platform_id* platforms;
    cl_device_id* devices;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_float* p_x;
    cl_float* p_y;
    cl_float* p_ret;
    cl_float* p_cl_ret;
    cl_float flt;
    cl_mem buf_x;
    cl_mem buf_y;
    time_t t1, t2;
    int ret, i;

    // There is practically no error-checking. Live life on the edge

    num_platforms = 1;
    platforms = malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);

    num_devices = 1;
    devices = malloc(num_devices * sizeof(cl_device_id));
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, NULL);
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    ret = clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        puts("clBuildProgram failed");
        char log[1<<16];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 1<<16, log, NULL);
        return 1;
    }

    kernel = clCreateKernel(program, "saxpy", NULL);

    p_x      = calloc(LENGTH, sizeof(cl_float));
    p_y      = calloc(LENGTH, sizeof(cl_float));
    p_ret    = calloc(LENGTH, sizeof(cl_float));
    p_cl_ret = calloc(LENGTH, sizeof(cl_float));

    srand(3);
    for (int i = 0; i < LENGTH; i++) {
        p_x[i] = (cl_float)rand() / RAND_MAX;
        p_y[i] = (cl_float)rand() / RAND_MAX;
    }

    buf_x = clCreateBuffer(context, CL_MEM_READ_ONLY,  LENGTH * sizeof(cl_float), NULL, NULL);
    buf_y = clCreateBuffer(context, CL_MEM_READ_WRITE, LENGTH * sizeof(cl_float), NULL, NULL);

    flt = 2.0f;
    t1 = clock();
    t2 = 0;

    for (i = 0; i < LENGTH; i++)
        p_ret[i] = flt * p_x[i] + p_y[i];
    t1 = clock() - t1;

    printf("CPU: %f\n", (double)t1 / CLOCKS_PER_SEC);

    t1 = clock();
    clEnqueueWriteBuffer(queue, buf_x, CL_TRUE, 0, LENGTH * sizeof(cl_float), p_x, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_y, CL_TRUE, 0, LENGTH * sizeof(cl_float), p_y, 0, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_y);
    clSetKernelArg(kernel, 2, sizeof(cl_float), &flt);
    t1 = clock() - t1;
    t2 += t1;
    printf("GPU (init): %f\n", (double)t1 / CLOCKS_PER_SEC);

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (size_t*)&LENGTH, NULL, 0, NULL, NULL);
    clFinish(queue);

    t1 = clock() - t1;
    t2 += t1;
    printf("GPU (run): %f\n", (double)t1 / CLOCKS_PER_SEC);

    clEnqueueReadBuffer(queue, buf_y, CL_TRUE, 0, LENGTH * sizeof(cl_float), p_cl_ret, 0, NULL, NULL);
    t1 = clock() - t1;
    t2 += t1;
    printf("GPU (read): %f\n", (double)t1 / CLOCKS_PER_SEC);
    printf("GPU (total): %f\n", (double)t1 / CLOCKS_PER_SEC);

    for (i = 0; i < LENGTH; i++) {
        if (fabs(p_ret[i] - p_cl_ret[i]) > 0.0001) {
            puts("failed");
            break;
        }
    }

    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_y);
    free(p_x);
    free(p_y);
    free(p_ret);
    free(p_cl_ret);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);
    free(devices);
    return 0;
}
