#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/cl.h>
#include <time.h>
#include <util.h>

void print_info(cl_platform_id platform, cl_device_id device)
{
    char* value;
    cl_uint num;
    size_t size;
    size_t sizes[5];

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
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL);
    printf("LOCAL DIM: %llu %llu %llu\n", sizes[0], sizes[1], sizes[2]);

    free(value);
}

void print_kernel_info(const char* name, cl_kernel kernel, cl_device_id device)
{
    size_t size;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
    puts(name);
    printf("KERNEL WORK GROUP SIZE: %llu\n", size);
}

void print_time(cl_int N, time_t t, const char* device, cl_float* C)
{
    printf("%s: %f\n", device, (double)(clock() - t)/CLOCKS_PER_SEC);
    printf("First 5:  %f %f %f %f %f\n", C[0], C[1], C[2], C[3], C[4]);
    printf("Middle 5: %f %f %f %f %f\n", C[N/2-2], C[N/2-1],C[N/2], C[N/2+1], C[N/2+2]);
    printf("Last 5:   %f %f %f %f %f\n", C[N-5], C[N-4], C[N-3], C[N-2], C[N-1]);
}

void test_cpu(cl_int N, cl_float* A, cl_float* B, cl_float* C)
{
    int i, j, k;
    time_t t;
    t = clock();

    // CPU
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];

    print_time(N, t, "CPU", C); 
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
    cl_kernel matmul;
    cl_kernel matmul_local;
    cl_kernel matmul_dim3;
    cl_mem cl_A, cl_B, cl_C1, cl_C2, cl_C3;
    char* source;
    cl_float* A;
    cl_float* B;
    cl_float* C;
    cl_int N;
    size_t size;
    size_t local_size;
    int ret, i;
    time_t t;

    num_platforms = 1;
    platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    num_devices = 1;
    devices = malloc(num_devices * sizeof(cl_device_id));
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    print_info(platforms[0], devices[0]);
    puts("");

    N = 32<<6;
    A = calloc(N * N, sizeof(cl_float));
    B = calloc(N * N, sizeof(cl_float));
    C = calloc(N * N, sizeof(cl_float));
    size = N * N * sizeof(cl_float);
    
    printf("N=%d\n", N);

    srand(1);
    for (i = 0; i < N*N; i++) {
        A[i] = (cl_float)rand() / RAND_MAX;
        B[i] = (cl_float)rand() / RAND_MAX;
    }

    //test_cpu(N, A, B, C);

    size_t global_work_dim[3] = {N, N, N};

    size_t local_work_dim[2] = {32, 32};
    local_size = local_work_dim[0] * local_work_dim[1] * sizeof(cl_float);

    context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, NULL);
    source = read_file("matmul.kern");
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);
    ret = clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        puts("clBuildProgram failed");
        char log[1<<16];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 1<<16, log, NULL);
        puts(log);
        return 1;
    }

    matmul = clCreateKernel(program, "matmul", NULL);
    matmul_local = clCreateKernel(program, "matmul_local", NULL);
    matmul_dim3 = clCreateKernel(program, "matmul_dim3", NULL);

    print_kernel_info("matmul", matmul, devices[0]);
    print_kernel_info("matmul_local", matmul_local, devices[0]);
    puts("");

    cl_A = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    cl_B = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    cl_C1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    cl_C2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    cl_C3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);

    t = clock();
    clEnqueueWriteBuffer(queue, cl_A, CL_FALSE, 0, N * N * sizeof(cl_float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cl_B, CL_FALSE, 0, N * N * sizeof(cl_float), B, 0, NULL, NULL);
    clFinish(queue);
    clSetKernelArg(matmul, 0, sizeof(cl_int), &N);
    clSetKernelArg(matmul, 1, sizeof(cl_mem), &cl_A);
    clSetKernelArg(matmul, 2, sizeof(cl_mem), &cl_B);
    clSetKernelArg(matmul, 3, sizeof(cl_mem), &cl_C1);
    ret = clEnqueueNDRangeKernel(queue, matmul, 2, NULL, global_work_dim,  NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(queue);
    clEnqueueReadBuffer(queue, cl_C1, CL_TRUE, 0, size, C, 0, NULL, NULL);
    clFinish(queue);
    print_time(N, t, "matmul", C);
    puts("");

    t = clock();
    clEnqueueWriteBuffer(queue, cl_A, CL_FALSE, 0, N * N * sizeof(cl_float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cl_B, CL_FALSE, 0, N * N * sizeof(cl_float), B, 0, NULL, NULL);
    clFinish(queue);
    clSetKernelArg(matmul_local, 0, sizeof(cl_int), &N);
    clSetKernelArg(matmul_local, 1, sizeof(cl_mem), &cl_A);
    clSetKernelArg(matmul_local, 2, sizeof(cl_mem), &cl_B);
    clSetKernelArg(matmul_local, 3, sizeof(cl_mem), &cl_C2);
    clSetKernelArg(matmul_local, 4, local_size, NULL);
    clSetKernelArg(matmul_local, 5, local_size, NULL);
    ret = clEnqueueNDRangeKernel(queue, matmul_local, 2, NULL, global_work_dim, local_work_dim, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(queue);
    clEnqueueReadBuffer(queue, cl_C2, CL_TRUE, 0, size, C, 0, NULL, NULL);
    clFinish(queue);
    print_time(N, t, "matmul_local", C);

    t = clock();
    clEnqueueWriteBuffer(queue, cl_A, CL_FALSE, 0, N * N * sizeof(cl_float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cl_B, CL_FALSE, 0, N * N * sizeof(cl_float), B, 0, NULL, NULL);
    clFinish(queue);
    clSetKernelArg(matmul_dim3, 0, sizeof(cl_int), &N);
    clSetKernelArg(matmul_dim3, 1, sizeof(cl_mem), &cl_A);
    clSetKernelArg(matmul_dim3, 2, sizeof(cl_mem), &cl_B);
    clSetKernelArg(matmul_dim3, 3, sizeof(cl_mem), &cl_C3);
    ret = clEnqueueNDRangeKernel(queue, matmul_dim3, 3, NULL, global_work_dim,  NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(queue);
    clEnqueueReadBuffer(queue, cl_C3, CL_TRUE, 0, size, C, 0, NULL, NULL);
    clFinish(queue);
    print_time(N, t, "matmul_dim3", C);
    puts("");

    clReleaseMemObject(cl_A);
    clReleaseMemObject(cl_B);
    clReleaseMemObject(cl_C1);
    clReleaseMemObject(cl_C2);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(source);
    free(platforms);
    free(devices);
    return 0;
}
