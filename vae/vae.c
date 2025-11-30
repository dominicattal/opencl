#include "vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/cl.h>
#include <util.h>

typedef enum {
    KERN_FORWARD,
    KERN_BACKWARD,
    KERN_BACKWARD_NO_UPDATE,
    KERN_ERROR_SET,
    NUM_KERNELS
} KernelEnum;

const char* kernel_names[NUM_KERNELS] = {
    "forward",
    "backward",
    "backward_no_update",
    "error_set",
};

typedef struct {
    cl_int num_rows;
    cl_int num_cols;
} MatrixSize;

typedef struct {
    cl_int*         layer_lengths;
    MatrixSize*     weight_sizes;
    cl_mem*         cl_buffers;
    cl_mem*         cl_weight_buffers;
    cl_int          num_layers;
    cl_int          input_length;
    cl_int          output_length;
} NeuralNet;

typedef struct VAE {

    // CL info
    cl_platform_id      platform;
    cl_device_id        device;
    cl_context          context;
    cl_command_queue    queue;
    cl_program          program;
    cl_kernel           kernels[NUM_KERNELS];

    // Model info
    cl_int              num_layers;
    cl_int              latent_space_length;
    cl_mem              cl_latent_space_buffer;
    cl_mem              cl_input_image_buffer;
    cl_mem              cl_output_image_buffer;
    cl_mem              cl_error_buffers[2];
    cl_float            learning_rate;
    NeuralNet           encoder;
    NeuralNet           decoder;

    // Image info
    cl_int width, height, num_pixels;

} VAE;

static cl_float rand_float(void)
{
    return (cl_float)(rand() - (RAND_MAX>>1)) / RAND_MAX * 2; // [-1, 1]
}

static void fill_buffer_random(cl_float* buffer, int num_rows, int num_cols)
{
    int i, j;
    for (i = 0; i < num_rows; i++)
        for (j = 0; j < num_cols; j++)
            buffer[i*num_cols+j] = rand_float();
}

static void read_info(cl_platform_id platform)
{
    char* value;
    size_t size;

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &size);
    value = malloc(size * sizeof(char));
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, value, NULL);
    puts(value);

    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &size);
    value = malloc(size * sizeof(char));
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, value, NULL);
    puts(value);

    free(value);
}

static void neuralnet_create(VAE* vae, NeuralNet* net, cl_int input_length, cl_int output_length)
{
    cl_float* buf;
    size_t size;
    cl_int i, n, max_length;
    cl_int num_rows, num_cols;

    n = net->num_layers = vae->num_layers;
    net->input_length = input_length;
    net->output_length = output_length;
    max_length = (input_length > output_length) ? input_length : output_length;

    for (i = 0; i < n; i++)
        max_length = (max_length > net->layer_lengths[i]) ? max_length : net->layer_lengths[i];

    buf = malloc((max_length+1) * (max_length+1) * sizeof(cl_float));

    net->weight_sizes = malloc((n+1) * sizeof(MatrixSize));
    net->cl_buffers = malloc(n * sizeof(cl_mem));
    net->cl_weight_buffers = malloc((n+1) * sizeof(cl_mem));
    for (i = 0; i < n; i++) {
        size = net->layer_lengths[i] * sizeof(cl_float);
        net->cl_buffers[i] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }

    num_rows = input_length+1; // add 1 for bias
    num_cols = net->layer_lengths[0];
    size = num_rows * num_cols * sizeof(cl_float);
    net->weight_sizes[0].num_rows = num_rows;
    net->weight_sizes[0].num_cols = num_cols;
    net->cl_weight_buffers[0] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    for (i = 1; i < n; i++) {
        num_rows = net->layer_lengths[i-1]+1;
        num_cols = net->layer_lengths[i];
        size = num_rows * num_cols * sizeof(cl_float);
        net->weight_sizes[i].num_rows = num_rows;
        net->weight_sizes[i].num_cols = num_cols;
        net->cl_weight_buffers[i] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }
    num_rows = net->layer_lengths[n-1]+1;
    num_cols = output_length;
    size = num_rows * num_cols * sizeof(cl_float);
    net->weight_sizes[i].num_rows = num_rows;
    net->weight_sizes[i].num_cols = num_cols;
    net->cl_weight_buffers[n] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);

    // randomize weights
    for (i = 0; i <= n; i++) {
        num_rows = net->weight_sizes[i].num_rows;
        num_cols = net->weight_sizes[i].num_cols;
        size = num_rows * num_cols * sizeof(cl_float);
        fill_buffer_random(buf, num_rows, num_cols);
        clEnqueueWriteBuffer(vae->queue, net->cl_weight_buffers[i], CL_TRUE, 0, size, buf, 0, NULL, NULL);
    }

    free(buf);
}

static void neuralnet_destroy(NeuralNet* net)
{
    int i;
    for (i = 0; i < net->num_layers; i++)
        clReleaseMemObject(net->cl_buffers[i]);
    for (i = 0; i <= net->num_layers; i++)
        clReleaseMemObject(net->cl_weight_buffers[i]);
    free(net->layer_lengths);
    free(net->cl_weight_buffers);
    free(net->cl_buffers);
    free(net->weight_sizes);
}

static void initialize_opencl(VAE* vae)
{
    char* source;
    cl_int ret;
    int i;
    clGetPlatformIDs(1, &vae->platform, NULL);
    clGetDeviceIDs(vae->platform, CL_DEVICE_TYPE_GPU, 1, &vae->device, NULL);
    vae->context = clCreateContext(NULL, 1, &vae->device, NULL, NULL, NULL);
    vae->queue = clCreateCommandQueueWithProperties(vae->context, vae->device, NULL, NULL);
    source = read_file("vae.kern");
    vae->program = clCreateProgramWithSource(vae->context, 1, (const char**)&source, NULL, NULL);
    free(source);
    ret = clBuildProgram(vae->program, 1, &vae->device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        puts("clBuildProgram failed");
        char log[1<<16];
        clGetProgramBuildInfo(vae->program, vae->device, CL_PROGRAM_BUILD_LOG, 1<<16, log, NULL);
        puts(log);
        exit(1);
    }
    for (i = 0; i < NUM_KERNELS; i++)
        vae->kernels[i] = clCreateKernel(vae->program, kernel_names[i], NULL);
    //read_info(vae->platform);

}

VAE* vae_create(int img_width, int img_height, int latent_space_length, int num_layers, int* layer_lengths)
{
    size_t size;
    int i;
    VAE* vae;

    if (num_layers <= 1) {
        puts("Number of layers must be greater than 1");
        return NULL;
    }

    vae = malloc(sizeof(VAE));
    initialize_opencl(vae);

    vae->width = img_width;
    vae->height = img_height;
    vae->num_pixels = img_width * img_height;
    vae->num_layers = num_layers;
    vae->latent_space_length = latent_space_length;

    size = latent_space_length * sizeof(cl_float);
    vae->cl_latent_space_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    size = vae->num_pixels * sizeof(cl_float);
    vae->cl_input_image_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->cl_output_image_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->cl_error_buffers[0] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->cl_error_buffers[1] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);

    vae->encoder.layer_lengths = malloc(num_layers * sizeof(cl_int));
    vae->decoder.layer_lengths = malloc(num_layers * sizeof(cl_int));
    for (i = 0; i < num_layers; i++) {
        vae->encoder.layer_lengths[i] = layer_lengths[i];
        vae->decoder.layer_lengths[num_layers-i-1] = layer_lengths[i];
    }
    neuralnet_create(vae, &vae->encoder, vae->num_pixels, vae->latent_space_length);
    neuralnet_create(vae, &vae->decoder, vae->latent_space_length, vae->num_pixels);

    return vae;
}

void vae_destroy(VAE* vae)
{
    neuralnet_destroy(&vae->encoder);
    neuralnet_destroy(&vae->decoder);
    clReleaseMemObject(vae->cl_latent_space_buffer);
    clReleaseMemObject(vae->cl_input_image_buffer);
    clReleaseMemObject(vae->cl_output_image_buffer);
    clReleaseMemObject(vae->cl_error_buffers[0]);
    clReleaseMemObject(vae->cl_error_buffers[1]);
    clReleaseProgram(vae->program);
    clReleaseCommandQueue(vae->queue);
    clReleaseContext(vae->context);
}

static void neuralnet_feedforward(VAE* vae, NeuralNet* nn, cl_mem cl_buf_in, cl_int cl_buf_in_length, cl_mem cl_buf_out, cl_int cl_buf_out_length)
{
    cl_event* event_wait_list;
    cl_event event;
    cl_kernel kernel;
    size_t work_dim_size;
    int i, n, num_events, ret;
    n = vae->num_layers;
    kernel = vae->kernels[KERN_FORWARD];
    event_wait_list = malloc((n + 1) * sizeof(cl_event));
    num_events = 0;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[0]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[0]);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &cl_buf_in_length);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &nn->layer_lengths[0]);
    work_dim_size = nn->layer_lengths[0];
    ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_dim_size, 
                           NULL, 0, NULL, &event);
    assert(ret == CL_SUCCESS);
    event_wait_list[num_events++] = event;
    for (i = 1; i < n; i++) {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[i-1]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[i]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[i]);
        clSetKernelArg(kernel, 3, sizeof(cl_int), &nn->layer_lengths[i-1]);
        clSetKernelArg(kernel, 4, sizeof(cl_int), &nn->layer_lengths[i]);
        work_dim_size = nn->layer_lengths[i];
        ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_dim_size, 
                            NULL, num_events, event_wait_list, &event);
        assert(ret == CL_SUCCESS);
        event_wait_list[num_events++] = event;
    }
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[n-1]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buf_out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[n]);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &nn->layer_lengths[n-1]);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &cl_buf_out_length);
    work_dim_size = cl_buf_out_length;
    ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_dim_size, 
                           NULL, num_events, event_wait_list, &event);
    assert(ret == CL_SUCCESS);
    clFinish(vae->queue);
    free(event_wait_list);
}

static void neuralnet_backpropagate(VAE* vae, NeuralNet* nn, cl_mem cl_buf_in, cl_int cl_buf_in_length, cl_mem cl_buf_out, cl_int cl_buf_out_length, bool do_updates)
{
    cl_event* event_wait_list;
    cl_event event;
    cl_kernel kernel;
    size_t work_dim_size;
    cl_mem tmp;
    int i, j, n, num_events, ret;
    n = vae->num_layers;
    if (do_updates)
        kernel = vae->kernels[KERN_BACKWARD];
    else
        kernel = vae->kernels[KERN_BACKWARD_NO_UPDATE];
    event_wait_list = malloc((n + 1) * sizeof(cl_event));
    num_events = 0;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[n-1]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buf_out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[1]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &vae->cl_error_buffers[0]);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[n]);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &nn->layer_lengths[n-1]);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &cl_buf_out_length);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &vae->learning_rate);
    work_dim_size = nn->layer_lengths[n-1]+1;
    ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_dim_size,
                            NULL, 0, NULL, &event);
    clFinish(vae->queue);
    assert(ret == CL_SUCCESS);
    event_wait_list[num_events++] = event;
    for (i = n-2; i >= 0; i--) {
        j = (n-i)%2;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[i]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[i+1]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[j]);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &vae->cl_error_buffers[1-j]);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[i+1]);
        clSetKernelArg(kernel, 5, sizeof(cl_int), &nn->layer_lengths[i]);
        clSetKernelArg(kernel, 6, sizeof(cl_int), &nn->layer_lengths[i+1]);
        clSetKernelArg(kernel, 7, sizeof(cl_int), &vae->learning_rate);
        work_dim_size = nn->layer_lengths[i]+1;
        ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_dim_size,
                                NULL, num_events, event_wait_list, &event);
        assert(ret == CL_SUCCESS);
        event_wait_list[num_events++] = event;
    }
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[0]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[1-(n%2)]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &vae->cl_error_buffers[n%2]);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[0]);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &cl_buf_in_length);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &nn->layer_lengths[0]);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &vae->learning_rate);
    work_dim_size = cl_buf_in_length+1;
    ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_dim_size,
                            NULL, num_events, event_wait_list, &event);
    assert(ret == CL_SUCCESS);
    clFinish(vae->queue);
    free(event_wait_list);

    // ensure last error is in correct spot
    if (n % 2 == 0) {
        tmp = vae->cl_error_buffers[0];
        vae->cl_error_buffers[0] = vae->cl_error_buffers[1];
        vae->cl_error_buffers[1] = tmp;
    }
}

static void feedforward(VAE* vae)
{
    neuralnet_feedforward(vae, &vae->encoder, vae->cl_input_image_buffer, vae->num_pixels,
            vae->cl_latent_space_buffer, vae->latent_space_length);
    neuralnet_feedforward(vae, &vae->decoder, vae->cl_latent_space_buffer, vae->latent_space_length, 
            vae->cl_output_image_buffer, vae->num_pixels);
}

static void backpropagate(VAE* vae)
{
    neuralnet_backpropagate(vae, &vae->decoder, vae->cl_latent_space_buffer, vae->latent_space_length, 
            vae->cl_output_image_buffer, vae->num_pixels, true);
    neuralnet_backpropagate(vae, &vae->encoder, vae->cl_input_image_buffer, vae->num_pixels,
            vae->cl_latent_space_buffer, vae->latent_space_length, true);
}

void vae_train(VAE* vae, int num_images, float** image_data, float learning_rate, int epochs)
{
    cl_float mse;
    cl_float* data;
    cl_kernel kernel;
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    size_t work_size_dim;
    int i, k, epoch;

    data = malloc(vae->num_pixels * sizeof(cl_float));
    vae->learning_rate = learning_rate;
    work_size_dim = vae->num_pixels+1;
    for (epoch = 0; epoch < epochs; epoch++) {
        for (i = 0; i < num_images; i++) {
            clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, image_data[i], 0, NULL, NULL);
            feedforward(vae);
            kernel = vae->kernels[KERN_ERROR_SET];
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &vae->cl_input_image_buffer);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &vae->cl_output_image_buffer);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[0]);
            clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_size_dim, NULL, 0, NULL, NULL);
            clFinish(vae->queue);
            if (i%1000 == 0) {
                clEnqueueReadBuffer(vae->queue, vae->cl_error_buffers[0], CL_TRUE, 0, img_size, data, 0, NULL, NULL);
                mse = 0;
                for (k = 0; k < vae->num_pixels; k++)
                    mse += 0.5 * data[k] * data[k];
                printf("%d %f\n", i, mse);
            }
            backpropagate(vae);
        }
    }
    free(data);
}

float* vae_feedforward(VAE* vae, float* data)
{
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    float* output = malloc(img_size * sizeof(float));
    clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    feedforward(vae);
    clEnqueueReadBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    return output;
}

float* vae_encode(VAE* vae, float* data)
{
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    size_t latent_space_size = vae->latent_space_length * sizeof(cl_float);
    float* output = malloc(vae->latent_space_length * sizeof(float));
    clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    neuralnet_feedforward(vae, &vae->encoder, vae->cl_input_image_buffer, vae->num_pixels,
            vae->cl_latent_space_buffer, vae->latent_space_length);
    clEnqueueReadBuffer(vae->queue, vae->cl_latent_space_buffer, CL_TRUE, 0, latent_space_size, output, 0, NULL, NULL);
    return output;
}

float* vae_decode(VAE* vae, float* data)
{
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    size_t latent_space_size = vae->latent_space_length * sizeof(cl_float);
    float* output = malloc(img_size * sizeof(float));
    clEnqueueWriteBuffer(vae->queue, vae->cl_latent_space_buffer, CL_TRUE, 0, latent_space_size, data, 0, NULL, NULL);
    neuralnet_feedforward(vae, &vae->decoder, vae->cl_latent_space_buffer, vae->latent_space_length, 
            vae->cl_output_image_buffer, vae->num_pixels);
    clEnqueueReadBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    return output;
}

float* vae_create_heatmap(VAE* vae, float* in_data, float* out_data)
{
    cl_float min_val, max_val, diff;
    cl_kernel kernel;
    size_t img_size, work_size_dim;
    float* output;
    int i;
    img_size = vae->num_pixels * sizeof(cl_float);
    output = malloc(img_size * sizeof(float));
    work_size_dim = vae->num_pixels+1;
    clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, in_data, 0, NULL, NULL);
    clEnqueueWriteBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, out_data, 0, NULL, NULL);
    kernel = vae->kernels[KERN_ERROR_SET];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vae->cl_input_image_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vae->cl_output_image_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[0]);
    clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_size_dim, NULL, 0, NULL, NULL);
    clFinish(vae->queue);
    neuralnet_backpropagate(vae, &vae->decoder, vae->cl_latent_space_buffer, vae->latent_space_length, 
            vae->cl_output_image_buffer, vae->num_pixels, false);
    neuralnet_backpropagate(vae, &vae->encoder, vae->cl_input_image_buffer, vae->num_pixels,
            vae->cl_latent_space_buffer, vae->latent_space_length, false);
    clEnqueueReadBuffer(vae->queue, vae->cl_error_buffers[0], CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    min_val = max_val = output[0];
    for (i = 1; i < vae->num_pixels; i++) {
        max_val = (output[i] > max_val) ? output[i] : max_val;
        min_val = (output[i] < min_val) ? output[i] : min_val;
    }
    printf("%f %f\n", min_val, max_val);
    diff = max_val - min_val;
    for (i = 0; i < vae->num_pixels; i++)
        output[i] = (output[i] - min_val) / diff;
    return output;
}

/* ===============================================================================
FORMAT: 
width                               -> 1 int
height                              -> 1 int
num_layers                          -> 1 int
latent_space_length                 -> 1 int
neural_nets                         -> 2
    input_length                    -> 1 int
    output_length                   -> 1 int
    layer_lengths                   -> num_layers ints
    weights                         -> num_layer + 1
        rows                        -> 1 int
        cols                        -> 1 int
        numbers                     -> (rows+1) * cols floats
*/

VAE* vae_read(const char* filename)
{
    FILE* fptr;
    cl_float* buf;
    cl_int max_length, length, i;
    size_t size;
    VAE* vae;

    vae = malloc(sizeof(VAE));
    initialize_opencl(vae);

    fptr = fopen(filename, "rb");

    fread(&vae->width, sizeof(cl_int), 1, fptr);
    fread(&vae->height, sizeof(cl_int), 1, fptr);
    vae->num_pixels = vae->width * vae->height;
    max_length = vae->num_pixels + 1;
    buf = malloc(max_length * max_length * sizeof(cl_float));
    size = vae->num_pixels * sizeof(cl_float);
    vae->cl_input_image_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->cl_output_image_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->cl_error_buffers[0] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->cl_error_buffers[1] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    fread(&vae->num_layers, sizeof(cl_int), 1, fptr);
    vae->encoder.num_layers = vae->num_layers;
    vae->decoder.num_layers = vae->num_layers;
    vae->encoder.layer_lengths = malloc(vae->num_layers * sizeof(cl_int));
    vae->decoder.layer_lengths = malloc(vae->num_layers * sizeof(cl_int));
    vae->encoder.cl_buffers = malloc(vae->num_layers * sizeof(cl_mem));
    vae->decoder.cl_buffers = malloc(vae->num_layers * sizeof(cl_mem));
    vae->encoder.weight_sizes = malloc((vae->num_layers+1) * sizeof(MatrixSize));
    vae->decoder.weight_sizes = malloc((vae->num_layers+1) * sizeof(MatrixSize));
    vae->encoder.cl_weight_buffers = malloc((vae->num_layers+1) * sizeof(cl_mem));
    vae->decoder.cl_weight_buffers = malloc((vae->num_layers+1) * sizeof(cl_mem));
    fread(&vae->latent_space_length, sizeof(cl_int), 1, fptr);
    size = vae->latent_space_length * sizeof(cl_float);
    vae->cl_latent_space_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);

    fread(&vae->encoder.input_length, sizeof(cl_int), 1, fptr);
    fread(&vae->encoder.output_length, sizeof(cl_int), 1, fptr);
    fread(vae->encoder.layer_lengths, sizeof(cl_int), vae->num_layers, fptr);
    for (i = 0; i < vae->num_layers; i++) {
        size = vae->encoder.layer_lengths[i] * sizeof(cl_float);
        vae->encoder.cl_buffers[i] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }
    for (i = 0; i <= vae->num_layers; i++) {
        fread(&vae->encoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fread(&vae->encoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        length = vae->encoder.weight_sizes[i].num_rows * vae->encoder.weight_sizes[i].num_cols;
        size = length * sizeof(cl_float);
        fread(buf, sizeof(cl_float), length, fptr);
        vae->encoder.cl_weight_buffers[i] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
        clEnqueueWriteBuffer(vae->queue, vae->encoder.cl_weight_buffers[i], CL_TRUE, 0,
                size, buf, 0, NULL, NULL);
    }

    fread(&vae->decoder.input_length, sizeof(cl_int), 1, fptr);
    fread(&vae->decoder.output_length, sizeof(cl_int), 1, fptr);
    fread(vae->decoder.layer_lengths, sizeof(cl_int), vae->num_layers, fptr);
    for (i = 0; i < vae->num_layers; i++) {
        size = vae->decoder.layer_lengths[i] * sizeof(cl_float);
        vae->decoder.cl_buffers[i] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }
    for (i = 0; i <= vae->num_layers; i++) {
        fread(&vae->decoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fread(&vae->decoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        length = vae->decoder.weight_sizes[i].num_rows * vae->decoder.weight_sizes[i].num_cols;
        size = length * sizeof(cl_float);
        fread(buf, sizeof(cl_float), length, fptr);
        vae->decoder.cl_weight_buffers[i] = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
        clEnqueueWriteBuffer(vae->queue, vae->decoder.cl_weight_buffers[i], CL_TRUE, 0,
                size, buf, 0, NULL, NULL);
    }
    return vae;
}

void vae_write(VAE* vae, const char* filename)
{
    int i;
    cl_int max_length, length, ret;
    FILE* fptr;
    cl_float* buf;

    max_length = vae->num_pixels + 1;
    buf = malloc(max_length * max_length * sizeof(cl_float));
    fptr = fopen(filename, "wb");
    fwrite(&vae->width, sizeof(cl_int), 1, fptr);
    fwrite(&vae->height, sizeof(cl_int), 1, fptr);
    fwrite(&vae->num_layers, sizeof(cl_int), 1, fptr);
    fwrite(&vae->latent_space_length, sizeof(cl_int), 1, fptr);

    fwrite(&vae->encoder.input_length, sizeof(cl_int), 1, fptr);
    fwrite(&vae->encoder.output_length, sizeof(cl_int), 1, fptr);
    fwrite(vae->encoder.layer_lengths, sizeof(cl_int), vae->num_layers, fptr);
    for (i = 0; i <= vae->num_layers; i++) {
        length = vae->encoder.weight_sizes[i].num_rows * vae->encoder.weight_sizes[i].num_cols;
        fwrite(&vae->encoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fwrite(&vae->encoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        ret = clEnqueueReadBuffer(vae->queue, vae->encoder.cl_weight_buffers[i], CL_TRUE, 0, 
                length * sizeof(cl_float), buf, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);
        fwrite(buf, sizeof(cl_float), length, fptr);
    }

    fwrite(&vae->decoder.input_length, sizeof(cl_int), 1, fptr);
    fwrite(&vae->decoder.output_length, sizeof(cl_int), 1, fptr);
    fwrite(vae->decoder.layer_lengths, sizeof(cl_int), vae->num_layers, fptr);
    for (i = 0; i <= vae->num_layers; i++) {
        length = vae->decoder.weight_sizes[i].num_rows * vae->decoder.weight_sizes[i].num_cols;
        fwrite(&vae->decoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fwrite(&vae->decoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        clEnqueueReadBuffer(vae->queue, vae->decoder.cl_weight_buffers[i], CL_TRUE, 0, 
                length * sizeof(cl_float), buf, 0, NULL, NULL);
        fwrite(buf, sizeof(cl_float), length, fptr);
    }

    free(buf);
}
