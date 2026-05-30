#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <util.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

const int WIDTH = 256;
const int HEIGHT = 256;
const int CHANNELS = 3;

void print_info(cl_platform_id platform, cl_device_id device)
{
    char* value;
    cl_uint num;
    size_t size;

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &size);
    value = malloc(size * sizeof(char));
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, value, NULL);
    puts(value);

    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &size);
    value = malloc(size * sizeof(char));
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, value, NULL);
    puts(value);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
    printf("WORK ITEM DIM: %d\n", num);

    free(value);
}

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
    cl_mem buf;
    unsigned char* data;
    char* source;
    size_t size;
    size_t global_work_size[2] = {WIDTH, HEIGHT};
    int ret;

    num_platforms = 1;
    platforms = malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);

    num_devices = 1;
    devices = malloc(num_devices * sizeof(cl_device_id));
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    print_info(platforms[0], devices[0]);

    context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, NULL);
    source = read_file("image.kern");
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);
    ret = clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        puts("clBuildProgram failed");
        char log[1<<16];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 1<<16, log, NULL);
        puts(log);
        return 1;
    }

    kernel = clCreateKernel(program, "create_image", NULL);
    data = calloc(WIDTH * HEIGHT * CHANNELS, sizeof(char));

    size = WIDTH * HEIGHT * CHANNELS * sizeof(cl_uchar);
    buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &WIDTH);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &HEIGHT);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &CHANNELS);
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(queue);

    ret = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, size, data, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);

    stbi_write_png("output.png", WIDTH, HEIGHT, CHANNELS, data, 0);

    clReleaseMemObject(buf);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(data);
    free(source);
    free(platforms);
    free(devices);
    return 0;
}
