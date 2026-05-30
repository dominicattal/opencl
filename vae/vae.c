#include "vae.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <util.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

typedef enum {
    KERN_FORWARD,
    KERN_FORWARD_SIGMOID,
    KERN_FORWARD_RELU,
    KERN_BACKWARD,
    KERN_BACKWARD_SIGMOID,
    KERN_BACKWARD_RELU,
    KERN_SAMPLE_DISTRIBUTIONS,
    KERN_SET_RECON_LOSS,
    KERN_ADD_KL_LOSS,
    NUM_KERNELS
} KernelEnum;

const char* kernel_names[NUM_KERNELS] = {
    "forward",
    "forward_sigmoid",
    "forward_relu",
    "backward",
    "backward_sigmoid",
    "backward_relu",
    "sample_distributions",
    "set_recon_loss",
    "add_kl_loss"
};

typedef struct {
    cl_int num_rows;
    cl_int num_cols;
} MatrixSize;

typedef struct {
    cl_int*         layer_lengths;
    ActivationEnum* layer_activations;
    MatrixSize*     weight_sizes;
    cl_mem*         cl_buffers;
    cl_mem*         cl_weight_buffers;
    cl_int          num_layers;
    cl_int          input_length;
    cl_int          output_length;
} NeuralNet;

typedef struct AutoEncoder {

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

} AutoEncoder;

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

    struct {
        cl_int              length;
        cl_mem              cl_sample_buffer;
        cl_mem              cl_distribution_buffer; // holds mean and logvar
        cl_mem              cl_epsilon_buffer;
    }                   latent_space;

    cl_mem              cl_input_image_buffer;
    cl_mem              cl_output_image_buffer;
    cl_mem              cl_error_buffers[2];
    cl_float            learning_rate;
    cl_float            beta;
    NeuralNet           encoder;
    NeuralNet           decoder;

    // Image info
    cl_int width, height, num_pixels;

} VAE;

// debugging
static void print_buf(cl_command_queue q, cl_mem mem, int length)
{
    cl_float* buf = malloc(length*sizeof(cl_float));
    clEnqueueReadBuffer(q, mem, CL_TRUE, 0, length*sizeof(cl_float), buf, 0, NULL, NULL);
    for (int i = 0; i < length; i++)
        printf("%f ", buf[i]);
    puts("");
    free(buf);
}


static cl_float rand_float(float min, float max)
{
    return (cl_float)(rand()) / RAND_MAX * (max-min) + min;
}

static cl_float guass_dist(float mean, float std)
{
    cl_float u1, u2, z;
    u1 = rand_float(0.001f, 1.0f); // cant be 0 or else outside of domain of log
    u2 = rand_float(0.0f, 1.0f);
    z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    return z * std + mean;
}

static void fill_buffer_random(cl_float* buffer, int num_rows, int num_cols, cl_float min, cl_float max)
{
    int i, j;
    for (i = 0; i < num_rows; i++)
        for (j = 0; j < num_cols; j++)
            buffer[i*num_cols+j] = rand_float(min, max);
}

static void neuralnet_create(cl_context context, cl_command_queue queue, NeuralNet* net, cl_int num_layers, cl_int input_length, cl_int output_length)
{
    cl_float* buf;
    size_t size;
    cl_int i, n, max_length;
    cl_int num_rows, num_cols;

    n = net->num_layers = num_layers;
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
        net->cl_buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }

    num_rows = input_length+1; // add 1 for bias
    num_cols = net->layer_lengths[0];
    size = num_rows * num_cols * sizeof(cl_float);
    net->weight_sizes[0].num_rows = num_rows;
    net->weight_sizes[0].num_cols = num_cols;
    net->cl_weight_buffers[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    for (i = 1; i < n; i++) {
        num_rows = net->layer_lengths[i-1]+1;
        num_cols = net->layer_lengths[i];
        size = num_rows * num_cols * sizeof(cl_float);
        net->weight_sizes[i].num_rows = num_rows;
        net->weight_sizes[i].num_cols = num_cols;
        net->cl_weight_buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }
    num_rows = net->layer_lengths[n-1]+1;
    num_cols = output_length;
    size = num_rows * num_cols * sizeof(cl_float);
    net->weight_sizes[i].num_rows = num_rows;
    net->weight_sizes[i].num_cols = num_cols;
    net->cl_weight_buffers[n] = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

    // randomize weights
    for (i = 0; i <= n; i++) {
        num_rows = net->weight_sizes[i].num_rows;
        num_cols = net->weight_sizes[i].num_cols;
        size = num_rows * num_cols * sizeof(cl_float);
        fill_buffer_random(buf, num_rows, num_cols, -0.1, 0.1);
        clEnqueueWriteBuffer(queue, net->cl_weight_buffers[i], CL_TRUE, 0, size, buf, 0, NULL, NULL);
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
    free(net->layer_activations);
    free(net->layer_lengths);
    free(net->cl_weight_buffers);
    free(net->cl_buffers);
    free(net->weight_sizes);
}

static void neuralnet_feedforward(cl_command_queue  queue, 
                                  cl_kernel*        kernels, 
                                  NeuralNet*        nn, 
                                  cl_mem            cl_buf_in, 
                                  cl_int            cl_buf_in_length, 
                                  cl_mem            cl_buf_out, 
                                  cl_int            cl_buf_out_length)
{
    cl_event* event_wait_list;
    cl_event event;
    cl_kernel kernel;
    size_t work_dim_size;
    int i, n, num_events, ret;
    ActivationEnum act_id;
    KernelEnum kern_id;

    static KernelEnum activation_to_kernel[NUM_ACTIVATIONS] = {
        KERN_FORWARD,
        KERN_FORWARD_SIGMOID,
        KERN_FORWARD_RELU
    };

    n = nn->num_layers;
    act_id = nn->layer_activations[0];
    kern_id = activation_to_kernel[act_id];
    kernel = kernels[kern_id];
    event_wait_list = malloc((n + 1) * sizeof(cl_event));
    num_events = 0;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[0]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[0]);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &cl_buf_in_length);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &nn->layer_lengths[0]);
    work_dim_size = nn->layer_lengths[0];
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim_size, NULL, 0, NULL, &event);
    assert(ret == CL_SUCCESS);
    event_wait_list[num_events++] = event;
    for (i = 1; i < n; i++) {
        act_id = nn->layer_activations[i];
        kern_id = activation_to_kernel[act_id];
        kernel = kernels[kern_id];
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[i-1]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[i]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[i]);
        clSetKernelArg(kernel, 3, sizeof(cl_int), &nn->layer_lengths[i-1]);
        clSetKernelArg(kernel, 4, sizeof(cl_int), &nn->layer_lengths[i]);
        work_dim_size = nn->layer_lengths[i];
        ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim_size, NULL, num_events, event_wait_list, &event);
        assert(ret == CL_SUCCESS);
        event_wait_list[num_events++] = event;
    }
    act_id = nn->layer_activations[n];
    kern_id = activation_to_kernel[act_id];
    kernel = kernels[kern_id];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[n-1]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buf_out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[n]);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &nn->layer_lengths[n-1]);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &cl_buf_out_length);
    work_dim_size = cl_buf_out_length;
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim_size, NULL, num_events, event_wait_list, &event);
    assert(ret == CL_SUCCESS);
    clFinish(queue);
    free(event_wait_list);
}

static void neuralnet_backpropagate(cl_command_queue    queue, 
                                    cl_kernel*          kernels, 
                                    NeuralNet*          nn, 
                                    cl_mem              cl_buf_in, 
                                    cl_int              cl_buf_in_length, 
                                    cl_mem              cl_buf_out, 
                                    cl_int              cl_buf_out_length, 
                                    cl_mem              cl_error_buffers[2],
                                    cl_float            learning_rate)
{
    cl_event* event_wait_list;
    cl_event event;
    cl_kernel kernel;
    size_t work_dim_size;
    cl_mem tmp;
    int i, j, n, num_events, ret;
    ActivationEnum act_id;
    KernelEnum kern_id;

    static KernelEnum activation_to_kernel[NUM_ACTIVATIONS] = {
        KERN_BACKWARD,
        KERN_BACKWARD_SIGMOID,
        KERN_BACKWARD_RELU
    };

    n = nn->num_layers;
    event_wait_list = malloc((n + 1) * sizeof(cl_event));
    num_events = 0;

    act_id = nn->layer_activations[n];
    kern_id = activation_to_kernel[act_id];
    kernel = kernels[kern_id];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[n-1]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buf_out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_error_buffers[1]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_error_buffers[0]);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[n]);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &nn->layer_lengths[n-1]);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &cl_buf_out_length);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &learning_rate);
    work_dim_size = nn->layer_lengths[n-1]+1;
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim_size, NULL, 0, NULL, &event);
    assert(ret == CL_SUCCESS);
    event_wait_list[num_events++] = event;
    for (i = n-2; i >= 0; i--) {
        j = (n-i)%2;
        act_id = nn->layer_activations[i+1];
        kern_id = activation_to_kernel[act_id];
        kernel = kernels[kern_id];
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[i]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[i+1]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_error_buffers[j]);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_error_buffers[1-j]);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[i+1]);
        clSetKernelArg(kernel, 5, sizeof(cl_int), &nn->layer_lengths[i]);
        clSetKernelArg(kernel, 6, sizeof(cl_int), &nn->layer_lengths[i+1]);
        clSetKernelArg(kernel, 7, sizeof(cl_int), &learning_rate);
        work_dim_size = nn->layer_lengths[i]+1;
        ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim_size,
                                NULL, num_events, event_wait_list, &event);
        assert(ret == CL_SUCCESS);
        event_wait_list[num_events++] = event;
    }
    act_id = nn->layer_activations[0];
    kern_id = activation_to_kernel[act_id];
    kernel = kernels[kern_id];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[0]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_error_buffers[1-(n%2)]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_error_buffers[n%2]);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[0]);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &cl_buf_in_length);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &nn->layer_lengths[0]);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &learning_rate);
    work_dim_size = cl_buf_in_length+1;
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim_size,
                            NULL, num_events, event_wait_list, &event);
    assert(ret == CL_SUCCESS);
    clFinish(queue);
    free(event_wait_list);

    // ensure last error is in correct spot
    if (n % 2 == 0) {
        tmp = cl_error_buffers[0];
        cl_error_buffers[0] = cl_error_buffers[1];
        cl_error_buffers[1] = tmp;
    }
}


static void initialize_opencl(cl_platform_id* platform, cl_device_id* device, cl_context* context, 
                              cl_command_queue* queue, cl_program* program, cl_kernel* kernels)
{
    char* source;
    cl_int ret;
    int i;
    clGetPlatformIDs(1, platform, NULL);
    clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
    *context = clCreateContext(NULL, 1, device, NULL, NULL, NULL);
    *queue = clCreateCommandQueueWithProperties(*context, *device, NULL, NULL);
    source = read_file("vae.kern");
    *program = clCreateProgramWithSource(*context, 1, (const char**)&source, NULL, NULL);
    free(source);
    ret = clBuildProgram(*program, 1, device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        puts("clBuildProgram failed");
        char log[1<<16];
        clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, 1<<16, log, NULL);
        puts(log);
        exit(1);
    }
    for (i = 0; i < NUM_KERNELS; i++)
        kernels[i] = clCreateKernel(*program, kernel_names[i], NULL);
    //read_info(*platform);

}

// ==============================================================================================
// Autoencoder implementation
// ==============================================================================================

AutoEncoder* ae_create(int img_width, int img_height, int latent_space_length, int num_layers, int* layer_lengths, ActivationEnum act)
{
    size_t size;
    int i;
    AutoEncoder* ae;

    if (num_layers <= 1) {
        puts("Number of layers must be greater than 1");
        return NULL;
    }

    ae = malloc(sizeof(AutoEncoder));
    initialize_opencl(&ae->platform, &ae->device, &ae->context, &ae->queue, &ae->program, ae->kernels);

    ae->width = img_width;
    ae->height = img_height;
    ae->num_pixels = img_width * img_height;
    ae->num_layers = num_layers;
    ae->latent_space_length = latent_space_length;

    size = latent_space_length * sizeof(cl_float);
    ae->cl_latent_space_buffer = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    size = ae->num_pixels * sizeof(cl_float);
    ae->cl_input_image_buffer = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    ae->cl_output_image_buffer = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    ae->cl_error_buffers[0] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    ae->cl_error_buffers[1] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);

    ae->encoder.layer_lengths = malloc(num_layers * sizeof(cl_int));
    ae->decoder.layer_lengths = malloc(num_layers * sizeof(cl_int));
    for (i = 0; i < num_layers; i++) {
        ae->encoder.layer_lengths[i] = layer_lengths[i];
        ae->decoder.layer_lengths[num_layers-i-1] = layer_lengths[i];
    }
    ae->encoder.layer_activations = malloc((num_layers+1) * sizeof(ActivationEnum));
    ae->decoder.layer_activations = malloc((num_layers+1) * sizeof(ActivationEnum));
    for (i = 0; i <= num_layers; i++) {
        ae->encoder.layer_activations[i] = act;
        ae->decoder.layer_activations[i] = act;
    }
    ae->encoder.layer_activations[num_layers] = ACT_NONE;
    ae->decoder.layer_activations[num_layers] = ACT_SIGMOID;
    neuralnet_create(ae->context, ae->queue, &ae->encoder, ae->num_layers, ae->num_pixels, ae->latent_space_length);
    neuralnet_create(ae->context, ae->queue, &ae->decoder, ae->num_layers, ae->latent_space_length, ae->num_pixels);

    return ae;
}

void ae_destroy(AutoEncoder* ae)
{
    neuralnet_destroy(&ae->encoder);
    neuralnet_destroy(&ae->decoder);
    clReleaseMemObject(ae->cl_latent_space_buffer);
    clReleaseMemObject(ae->cl_input_image_buffer);
    clReleaseMemObject(ae->cl_output_image_buffer);
    clReleaseMemObject(ae->cl_error_buffers[0]);
    clReleaseMemObject(ae->cl_error_buffers[1]);
    clReleaseProgram(ae->program);
    clReleaseCommandQueue(ae->queue);
    clReleaseContext(ae->context);
}

static void ae_train_feedforward(AutoEncoder* ae)
{
    neuralnet_feedforward(ae->queue, ae->kernels, &ae->encoder, ae->cl_input_image_buffer, 
            ae->num_pixels, ae->cl_latent_space_buffer, ae->latent_space_length);
    neuralnet_feedforward(ae->queue, ae->kernels, &ae->decoder, ae->cl_latent_space_buffer, 
            ae->latent_space_length, ae->cl_output_image_buffer, ae->num_pixels);
}

static void ae_train_backpropagate(AutoEncoder* ae)
{
    neuralnet_backpropagate(ae->queue, ae->kernels, &ae->decoder, ae->cl_latent_space_buffer, ae->latent_space_length, 
            ae->cl_output_image_buffer, ae->num_pixels, ae->cl_error_buffers, ae->learning_rate);
    neuralnet_backpropagate(ae->queue, ae->kernels, &ae->encoder, ae->cl_input_image_buffer, ae->num_pixels,
            ae->cl_latent_space_buffer, ae->latent_space_length, ae->cl_error_buffers, ae->learning_rate);
}

void ae_train(AutoEncoder* ae, int num_images, float** image_data, float learning_rate, int epochs)
{
    cl_float mse;
    cl_float* data;
    cl_kernel kernel;
    size_t img_size = ae->num_pixels * sizeof(cl_float);
    size_t work_size_dim;
    int i, k, epoch;
    clock_t start;

    start = clock();
    data = malloc(ae->num_pixels * sizeof(cl_float));
    ae->learning_rate = learning_rate;
    work_size_dim = ae->num_pixels+1;
    for (epoch = 0; epoch < epochs; epoch++) {
        for (i = 0; i < num_images; i++) {
            clEnqueueWriteBuffer(ae->queue, ae->cl_input_image_buffer, CL_TRUE, 0, img_size, image_data[i], 0, NULL, NULL);
            ae_train_feedforward(ae);
            kernel = ae->kernels[KERN_SET_RECON_LOSS];
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &ae->cl_input_image_buffer);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &ae->cl_output_image_buffer);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &ae->cl_error_buffers[0]);
            clEnqueueNDRangeKernel(ae->queue, kernel, 1, NULL, &work_size_dim, NULL, 0, NULL, NULL);
            clFinish(ae->queue);
            if (i%1000 == 0) {
                clEnqueueReadBuffer(ae->queue, ae->cl_error_buffers[0], CL_TRUE, 0, img_size, data, 0, NULL, NULL);
                mse = 0;
                for (k = 0; k < ae->num_pixels; k++)
                    mse += 0.5 * data[k] * data[k];
                printf("%d %f\n", i, mse);
            }
            ae_train_backpropagate(ae);
        }
    }
    free(data);
    printf("trained in %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);
}

float* ae_feedforward(AutoEncoder* ae, float* data)
{
    size_t img_size = ae->num_pixels * sizeof(cl_float);
    float* output = malloc(img_size * sizeof(float));
    clEnqueueWriteBuffer(ae->queue, ae->cl_input_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    ae_train_feedforward(ae);
    clEnqueueReadBuffer(ae->queue, ae->cl_output_image_buffer, CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    return output;
}

float* ae_encode(AutoEncoder* ae, float* data)
{
    size_t img_size = ae->num_pixels * sizeof(cl_float);
    size_t latent_space_size = ae->latent_space_length * sizeof(cl_float);
    float* output = malloc(ae->latent_space_length * sizeof(float));
    clEnqueueWriteBuffer(ae->queue, ae->cl_input_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    neuralnet_feedforward(ae->queue, ae->kernels, &ae->encoder, ae->cl_input_image_buffer, ae->num_pixels,
            ae->cl_latent_space_buffer, ae->latent_space_length);
    clEnqueueReadBuffer(ae->queue, ae->cl_latent_space_buffer, CL_TRUE, 0, latent_space_size, output, 0, NULL, NULL);
    return output;
}

float* ae_decode(AutoEncoder* ae, float* data)
{
    size_t img_size = ae->num_pixels * sizeof(cl_float);
    size_t latent_space_size = ae->latent_space_length * sizeof(cl_float);
    float* output = malloc(img_size * sizeof(float));
    clEnqueueWriteBuffer(ae->queue, ae->cl_latent_space_buffer, CL_TRUE, 0, latent_space_size, data, 0, NULL, NULL);
    neuralnet_feedforward(ae->queue, ae->kernels, &ae->decoder, ae->cl_latent_space_buffer, ae->latent_space_length, 
            ae->cl_output_image_buffer, ae->num_pixels);
    clEnqueueReadBuffer(ae->queue, ae->cl_output_image_buffer, CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    return output;
}

float** ae_create_heatmaps(AutoEncoder* ae, float* data)
{
    float** heatmaps;
    float* output;
    float* latent_space;
    float min_val, max_val, max_abs_val;
    float px;
    size_t img_size, ls_size;
    int i, j;

    img_size = ae->num_pixels * sizeof(float);
    ls_size = ae->latent_space_length * sizeof(float);
    latent_space = malloc(ls_size);
    output = calloc(ae->num_pixels, sizeof(float));
    heatmaps = malloc(ae->latent_space_length * sizeof(float*));
    for (i = 0; i < ae->latent_space_length; i++)
        heatmaps[i] = malloc(ae->num_pixels * sizeof(float));

    min_val = max_val = 0;
    clEnqueueWriteBuffer(ae->queue, ae->cl_output_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    for (i = 0; i < ae->num_pixels; i++) {
        output[i] = 1.0f;
        clEnqueueWriteBuffer(ae->queue, ae->cl_error_buffers[0], CL_TRUE, 0, img_size, output, 0, NULL, NULL);
        neuralnet_backpropagate(ae->queue, 
                                ae->kernels, 
                                &ae->decoder, 
                                ae->cl_latent_space_buffer,
                                ae->latent_space_length, 
                                ae->cl_output_image_buffer, 
                                ae->num_pixels, 
                                ae->cl_error_buffers, 
                                0.0f);
        clEnqueueReadBuffer(ae->queue, ae->cl_error_buffers[0], CL_TRUE, 0, ls_size, latent_space, 0, NULL, NULL);
        for (j = 0; j < ae->latent_space_length; j++) {
            heatmaps[j][i] = latent_space[j];
            min_val = (latent_space[j] < min_val) ? latent_space[j] : min_val;
            max_val = (latent_space[j] > max_val) ? latent_space[j] : max_val;
        }
        output[i] = 0.0f;
    }
    max_abs_val = (max_val > -min_val) ? max_val : -min_val;
    for (i = 0; i < ae->latent_space_length; i++) {
        for (j = 0; j < ae->num_pixels; j++) {
            px = heatmaps[i][j];
            if (px > 0)
                heatmaps[i][j] = fabs(px / max_abs_val) * 255;
            else
                heatmaps[i][j] = fabs(-px / max_abs_val) * 255;
            //if (px > 0)
            //    heatmaps[i][j] = px / max_val * 127.0 + 128.0;
            //else
            //    heatmaps[i][j] = 128 - px / min_val * 128.0;
        }
    }

    return heatmaps;
}

/*
AutoEncoder format: 
width                               -> 1 int
height                              -> 1 int
num_layers                          -> 1 int
latent_space_length                 -> 1 int
neural_nets                         -> 2
    input_length                    -> 1 int
    output_length                   -> 1 int
    layer_lengths                   -> num_layers ints
    activations                     -> num_layers + 1 ActivationEnum
    weights                         -> num_layer + 1
        rows                        -> 1 int
        cols                        -> 1 int
        numbers                     -> (rows+1) * cols floats
*/

AutoEncoder* ae_read(const char* filename)
{
    FILE* fptr;
    cl_float* buf;
    cl_int max_length, length, i;
    size_t size;
    AutoEncoder* ae;

    ae = malloc(sizeof(AutoEncoder));
    initialize_opencl(&ae->platform, &ae->device, &ae->context, &ae->queue, &ae->program, ae->kernels);

    fptr = fopen(filename, "rb");
    assert(fptr != NULL);

    fread(&ae->width, sizeof(cl_int), 1, fptr);
    fread(&ae->height, sizeof(cl_int), 1, fptr);
    ae->num_pixels = ae->width * ae->height;
    max_length = ae->num_pixels + 1;
    buf = malloc(max_length * max_length * sizeof(cl_float));
    size = ae->num_pixels * sizeof(cl_float);
    ae->cl_input_image_buffer = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    ae->cl_output_image_buffer = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    ae->cl_error_buffers[0] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    ae->cl_error_buffers[1] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    fread(&ae->num_layers, sizeof(cl_int), 1, fptr);
    ae->encoder.num_layers = ae->num_layers;
    ae->decoder.num_layers = ae->num_layers;
    ae->encoder.layer_lengths = malloc(ae->num_layers * sizeof(cl_int));
    ae->decoder.layer_lengths = malloc(ae->num_layers * sizeof(cl_int));
    ae->encoder.cl_buffers = malloc(ae->num_layers * sizeof(cl_mem));
    ae->decoder.cl_buffers = malloc(ae->num_layers * sizeof(cl_mem));
    ae->encoder.weight_sizes = malloc((ae->num_layers+1) * sizeof(MatrixSize));
    ae->decoder.weight_sizes = malloc((ae->num_layers+1) * sizeof(MatrixSize));
    ae->encoder.cl_weight_buffers = malloc((ae->num_layers+1) * sizeof(cl_mem));
    ae->decoder.cl_weight_buffers = malloc((ae->num_layers+1) * sizeof(cl_mem));
    ae->encoder.layer_activations = malloc((ae->num_layers+1) * sizeof(ActivationEnum));
    ae->decoder.layer_activations = malloc((ae->num_layers+1) * sizeof(ActivationEnum));
    fread(&ae->latent_space_length, sizeof(cl_int), 1, fptr);
    size = ae->latent_space_length * sizeof(cl_float);
    ae->cl_latent_space_buffer = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);

    fread(&ae->encoder.input_length, sizeof(cl_int), 1, fptr);
    fread(&ae->encoder.output_length, sizeof(cl_int), 1, fptr);
    fread(ae->encoder.layer_lengths, sizeof(cl_int), ae->num_layers, fptr);
    fread(ae->encoder.layer_activations, sizeof(ActivationEnum), ae->num_layers+1, fptr);
    for (i = 0; i < ae->num_layers; i++) {
        size = ae->encoder.layer_lengths[i] * sizeof(cl_float);
        ae->encoder.cl_buffers[i] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }
    for (i = 0; i <= ae->num_layers; i++) {
        fread(&ae->encoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fread(&ae->encoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        length = ae->encoder.weight_sizes[i].num_rows * ae->encoder.weight_sizes[i].num_cols;
        size = length * sizeof(cl_float);
        fread(buf, sizeof(cl_float), length, fptr);
        ae->encoder.cl_weight_buffers[i] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
        clEnqueueWriteBuffer(ae->queue, ae->encoder.cl_weight_buffers[i], CL_TRUE, 0,
                size, buf, 0, NULL, NULL);
    }

    fread(&ae->decoder.input_length, sizeof(cl_int), 1, fptr);
    fread(&ae->decoder.output_length, sizeof(cl_int), 1, fptr);
    fread(ae->decoder.layer_lengths, sizeof(cl_int), ae->num_layers, fptr);
    fread(ae->decoder.layer_activations, sizeof(ActivationEnum), ae->num_layers+1, fptr);
    for (i = 0; i < ae->num_layers; i++) {
        size = ae->decoder.layer_lengths[i] * sizeof(cl_float);
        ae->decoder.cl_buffers[i] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    }
    for (i = 0; i <= ae->num_layers; i++) {
        fread(&ae->decoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fread(&ae->decoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        length = ae->decoder.weight_sizes[i].num_rows * ae->decoder.weight_sizes[i].num_cols;
        size = length * sizeof(cl_float);
        fread(buf, sizeof(cl_float), length, fptr);
        ae->decoder.cl_weight_buffers[i] = clCreateBuffer(ae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
        clEnqueueWriteBuffer(ae->queue, ae->decoder.cl_weight_buffers[i], CL_TRUE, 0,
                size, buf, 0, NULL, NULL);
    }
    return ae;
}

void ae_write(AutoEncoder* ae, const char* filename)
{
    int i;
    cl_int max_length, length, ret;
    FILE* fptr;
    cl_float* buf;

    max_length = ae->num_pixels + 1;
    buf = malloc(max_length * max_length * sizeof(cl_float));
    fptr = fopen(filename, "wb");
    fwrite(&ae->width, sizeof(cl_int), 1, fptr);
    fwrite(&ae->height, sizeof(cl_int), 1, fptr);
    fwrite(&ae->num_layers, sizeof(cl_int), 1, fptr);
    fwrite(&ae->latent_space_length, sizeof(cl_int), 1, fptr);

    fwrite(&ae->encoder.input_length, sizeof(cl_int), 1, fptr);
    fwrite(&ae->encoder.output_length, sizeof(cl_int), 1, fptr);
    fwrite(ae->encoder.layer_lengths, sizeof(cl_int), ae->num_layers, fptr);
    fwrite(ae->encoder.layer_activations, sizeof(ActivationEnum), ae->num_layers+1, fptr);
    for (i = 0; i <= ae->num_layers; i++) {
        length = ae->encoder.weight_sizes[i].num_rows * ae->encoder.weight_sizes[i].num_cols;
        fwrite(&ae->encoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fwrite(&ae->encoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        ret = clEnqueueReadBuffer(ae->queue, ae->encoder.cl_weight_buffers[i], CL_TRUE, 0, 
                length * sizeof(cl_float), buf, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);
        fwrite(buf, sizeof(cl_float), length, fptr);
    }

    fwrite(&ae->decoder.input_length, sizeof(cl_int), 1, fptr);
    fwrite(&ae->decoder.output_length, sizeof(cl_int), 1, fptr);
    fwrite(ae->decoder.layer_lengths, sizeof(cl_int), ae->num_layers, fptr);
    fwrite(ae->decoder.layer_activations, sizeof(ActivationEnum), ae->num_layers+1, fptr);
    for (i = 0; i <= ae->num_layers; i++) {
        length = ae->decoder.weight_sizes[i].num_rows * ae->decoder.weight_sizes[i].num_cols;
        fwrite(&ae->decoder.weight_sizes[i].num_rows, sizeof(cl_int), 1, fptr);
        fwrite(&ae->decoder.weight_sizes[i].num_cols, sizeof(cl_int), 1, fptr);
        clEnqueueReadBuffer(ae->queue, ae->decoder.cl_weight_buffers[i], CL_TRUE, 0, 
                length * sizeof(cl_float), buf, 0, NULL, NULL);
        fwrite(buf, sizeof(cl_float), length, fptr);
    }

    free(buf);
}

int ae_get_latent_space_length(AutoEncoder* ae)
{
    return ae->latent_space_length;
}

// ==============================================================================================
// Autoencoder implementation end
// ==============================================================================================

// ==============================================================================================
// VAE implementation
// ==============================================================================================

VAE* vae_create(int img_width, int img_height, int latent_space_length, int num_layers, int* layer_lengths, ActivationEnum act)
{
    size_t size;
    int i;
    VAE* vae;

    if (num_layers <= 1) {
        puts("Number of layers must be greater than 1");
        return NULL;
    }

    vae = malloc(sizeof(VAE));
    initialize_opencl(&vae->platform, &vae->device, &vae->context, &vae->queue, &vae->program, vae->kernels);

    vae->width = img_width;
    vae->height = img_height;
    vae->num_pixels = img_width * img_height;
    vae->num_layers = num_layers;
    vae->latent_space.length = latent_space_length;

    size = latent_space_length * sizeof(cl_float);
    vae->latent_space.cl_sample_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->latent_space.cl_distribution_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, 2*size, NULL, NULL);
    vae->latent_space.cl_epsilon_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
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
    vae->encoder.layer_activations = malloc((num_layers+1) * sizeof(ActivationEnum));
    vae->decoder.layer_activations = malloc((num_layers+1) * sizeof(ActivationEnum));
    for (i = 0; i <= num_layers; i++) {
        vae->encoder.layer_activations[i] = act;
        vae->decoder.layer_activations[i] = act;
    }
    vae->encoder.layer_activations[num_layers] = ACT_NONE;
    neuralnet_create(vae->context, vae->queue, &vae->encoder, vae->num_layers, vae->num_pixels, 2*vae->latent_space.length);
    neuralnet_create(vae->context, vae->queue, &vae->decoder, vae->num_layers, vae->latent_space.length, vae->num_pixels);

    return vae;
}

void vae_destroy(VAE* vae)
{
    neuralnet_destroy(&vae->encoder);
    neuralnet_destroy(&vae->decoder);
    clReleaseMemObject(vae->latent_space.cl_sample_buffer);
    clReleaseMemObject(vae->latent_space.cl_distribution_buffer);
    clReleaseMemObject(vae->latent_space.cl_epsilon_buffer);
    clReleaseMemObject(vae->cl_input_image_buffer);
    clReleaseMemObject(vae->cl_output_image_buffer);
    clReleaseMemObject(vae->cl_error_buffers[0]);
    clReleaseMemObject(vae->cl_error_buffers[1]);
    clReleaseProgram(vae->program);
    clReleaseCommandQueue(vae->queue);
    clReleaseContext(vae->context);
}

static void vae_generate_epsilon_buffer(VAE* vae)
{
    // Box-Muller transform
    size_t size;
    cl_float* buf;
    buf = malloc(vae->latent_space.length * sizeof(cl_float));
    for (int i = 0; i < vae->latent_space.length; i++)
        buf[i] = guass_dist(0.0, 1.0);
    size = vae->latent_space.length * sizeof(cl_float);
    clEnqueueWriteBuffer(vae->queue, vae->latent_space.cl_epsilon_buffer, CL_TRUE, 0, size, buf, 0, NULL, NULL);
    free(buf);
}

void vae_seed(VAE* vae, unsigned long long seed)
{
    srand(seed);
    vae_generate_epsilon_buffer(vae);
}

static void vae_train_feedforward_encoder(VAE* vae)
{
    neuralnet_feedforward(vae->queue, 
                          vae->kernels, 
                          &vae->encoder, 
                          vae->cl_input_image_buffer, 
                          vae->num_pixels, 
                          vae->latent_space.cl_distribution_buffer, 
                          2*vae->latent_space.length);
}

static void vae_train_feedforward_decoder(VAE* vae)
{
    neuralnet_feedforward(vae->queue, 
                          vae->kernels, 
                          &vae->decoder, 
                          vae->latent_space.cl_sample_buffer, 
                          vae->latent_space.length, 
                          vae->cl_output_image_buffer, 
                          vae->num_pixels);
}

static void vae_train_sample_latent_space(VAE* vae)
{
    cl_int ret;
    cl_kernel kernel;
    size_t work_size_dim;
    kernel = vae->kernels[KERN_SAMPLE_DISTRIBUTIONS];
    work_size_dim = vae->latent_space.length;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vae->latent_space.cl_distribution_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vae->latent_space.cl_epsilon_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->latent_space.cl_sample_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &vae->latent_space.length);
    ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_size_dim, NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(vae->queue);
}

static void vae_train_feedforward(VAE* vae)
{
    vae_train_feedforward_encoder(vae);
    vae_train_sample_latent_space(vae);
    vae_train_feedforward_decoder(vae);
}

static void vae_train_backpropagate_encoder(VAE* vae)
{
    neuralnet_backpropagate(vae->queue, 
                            vae->kernels, 
                            &vae->encoder, 
                            vae->cl_input_image_buffer, 
                            vae->num_pixels, 
                            vae->latent_space.cl_distribution_buffer, 
                            2*vae->latent_space.length, 
                            vae->cl_error_buffers, 
                            vae->learning_rate);
}

static void vae_train_backpropagate_decoder(VAE* vae)
{
    neuralnet_backpropagate(vae->queue, 
                            vae->kernels, 
                            &vae->decoder, 
                            vae->latent_space.cl_sample_buffer, 
                            vae->latent_space.length, 
                            vae->cl_output_image_buffer, 
                            vae->num_pixels, 
                            vae->cl_error_buffers, 
                            vae->learning_rate);
}

static void vae_train_add_kl_loss(VAE* vae)
{
    cl_int ret;
    cl_kernel kernel;
    size_t work_size_dim;
    kernel = vae->kernels[KERN_ADD_KL_LOSS];
    work_size_dim = vae->latent_space.length;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vae->latent_space.cl_distribution_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vae->latent_space.cl_epsilon_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[0]);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &vae->latent_space.length);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &vae->beta);
    ret = clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_size_dim, NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(vae->queue);
}

static void vae_set_reconstruction_loss(VAE* vae)
{
    cl_kernel kernel;
    size_t work_size_dim;
    kernel = vae->kernels[KERN_SET_RECON_LOSS];
    work_size_dim = vae->num_pixels+1;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vae->cl_input_image_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vae->cl_output_image_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vae->cl_error_buffers[0]);
    clEnqueueNDRangeKernel(vae->queue, kernel, 1, NULL, &work_size_dim, NULL, 0, NULL, NULL);
    clFinish(vae->queue);
}

static void vae_train_backpropagate(VAE* vae)
{
    vae_set_reconstruction_loss(vae);
    vae_train_backpropagate_decoder(vae);
    vae_train_add_kl_loss(vae);
    vae_train_backpropagate_encoder(vae);
}

static void vae_print_losses(VAE* vae, float* data_in, int iter)
{
    cl_float* data_out;
    size_t img_size, dist_size, max_size;
    cl_float mse, kl_loss, diff;
    cl_float mean, logvar;
    int i;
    img_size = vae->num_pixels * sizeof(cl_float);
    dist_size = 2 * vae->latent_space.length * sizeof(cl_float);
    max_size = (img_size > dist_size) ? img_size : dist_size;
    data_out = malloc(max_size * sizeof(cl_float));
    clEnqueueReadBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, data_out, 0, NULL, NULL);
    mse = 0;
    for (i = 0; i < vae->num_pixels; i++) {
        diff = data_in[i] - data_out[i];
        mse += 0.5 * diff * diff;
    }
    clEnqueueReadBuffer(vae->queue, vae->latent_space.cl_distribution_buffer, CL_TRUE, 0, dist_size, data_out, 0, NULL, NULL);
    kl_loss = 0;
    for (i = 0; i < vae->latent_space.length; i++) {
        mean = data_out[i];
        logvar = data_out[i+vae->latent_space.length];
        kl_loss += 1 + logvar - mean*mean - exp(logvar);
    }
    kl_loss *= -0.5 * vae->beta;
    printf("%d mse=%f kl_loss=%f\n", iter, mse, kl_loss);
    free(data_out);
}

void vae_train(VAE* vae, int num_images, float** image_data, float learning_rate, float beta, int epochs)
{
    size_t img_size;
    int i, epoch;
    vae->learning_rate = learning_rate;
    vae->beta = beta;
    img_size = vae->num_pixels * sizeof(cl_float);
    for (epoch = 0; epoch < epochs; epoch++) {
        for (i = 0; i < num_images; i++) {
            vae_generate_epsilon_buffer(vae);
            clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, image_data[i], 0, NULL, NULL);
            vae_train_feedforward(vae);
            if (i % 1000 == 0)
                vae_print_losses(vae, image_data[i],  i);
            vae_train_backpropagate(vae);
        }
    }
}

float* vae_feedforward(VAE* vae, float* data)
{
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    float* output = malloc(img_size * sizeof(float));
    clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    vae_train_feedforward(vae);
    clEnqueueReadBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    return output;
}

float* vae_encode(VAE* vae, float* data)
{
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    size_t latent_space_size = vae->latent_space.length * sizeof(cl_float);
    float* output = malloc(vae->latent_space.length * sizeof(float));
    clEnqueueWriteBuffer(vae->queue, vae->cl_input_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    vae_train_feedforward_encoder(vae);
    vae_train_sample_latent_space(vae);
    clEnqueueReadBuffer(vae->queue, vae->latent_space.cl_sample_buffer, CL_TRUE, 0, latent_space_size, output, 0, NULL, NULL);
    return output;
}

float* vae_decode(VAE* vae, float* data)
{
    size_t img_size = vae->num_pixels * sizeof(cl_float);
    size_t latent_space_size = vae->latent_space.length * sizeof(cl_float);
    float* output = malloc(img_size * sizeof(float));
    clEnqueueWriteBuffer(vae->queue, vae->latent_space.cl_sample_buffer, CL_TRUE, 0, latent_space_size, data, 0, NULL, NULL);
    vae_train_feedforward_decoder(vae);
    clEnqueueReadBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, output, 0, NULL, NULL);
    return output;
}

float** vae_create_heatmaps(VAE* vae, float* data)
{
    float** heatmaps;
    float* output;
    float* latent_space;
    float min_val, max_val, max_abs_val;
    float px;
    size_t img_size, ls_size;
    int i, j;

    img_size = vae->num_pixels * sizeof(float);
    ls_size = vae->latent_space.length * sizeof(float);
    output = calloc(vae->num_pixels, sizeof(float));
    heatmaps = malloc(vae->latent_space.length * sizeof(float*));
    latent_space = malloc(ls_size);
    for (i = 0; i < vae->latent_space.length; i++)
        heatmaps[i] = malloc(vae->num_pixels * sizeof(float));

    min_val = max_val = 0;
    clEnqueueWriteBuffer(vae->queue, vae->cl_output_image_buffer, CL_TRUE, 0, img_size, data, 0, NULL, NULL);
    for (i = 0; i < vae->num_pixels; i++) {
        output[i] = 1.0f;
        clEnqueueWriteBuffer(vae->queue, vae->cl_error_buffers[0], CL_TRUE, 0, img_size, output, 0, NULL, NULL);
        neuralnet_backpropagate(vae->queue, 
                                vae->kernels, 
                                &vae->decoder, 
                                vae->latent_space.cl_sample_buffer, 
                                vae->latent_space.length, 
                                vae->cl_output_image_buffer, 
                                vae->num_pixels, 
                                vae->cl_error_buffers, 
                                0.0f);
        clEnqueueReadBuffer(vae->queue, vae->cl_error_buffers[0], CL_TRUE, 0, ls_size, latent_space, 0, NULL, NULL);
        for (j = 0; j < vae->latent_space.length; j++) {
            heatmaps[j][i] = latent_space[j];
            min_val = (latent_space[j] < min_val) ? latent_space[j] : min_val;
            max_val = (latent_space[j] > max_val) ? latent_space[j] : max_val;
        }
        output[i] = 0.0f;
    }
    max_abs_val = (max_val > -min_val) ? max_val : -min_val;
    for (i = 0; i < vae->latent_space.length; i++) {
        for (j = 0; j < vae->num_pixels; j++) {
            px = heatmaps[i][j];
            if (px > 0)
                heatmaps[i][j] = fabs(px / max_abs_val) * 255;
            else
                heatmaps[i][j] = fabs(-px / max_abs_val) * 255;
            //if (px > 0)
            //    heatmaps[i][j] = px / max_val * 127.0 + 128.0;
            //else
            //    heatmaps[i][j] = 128 - px / min_val * 128.0;
        }
    }
    free(output);
    free(latent_space);
    return heatmaps;
}

/*
VAE format: 
width                               -> 1 int
height                              -> 1 int
num_layers                          -> 1 int
latent_space_length                 -> 1 int
neural_nets                         -> 2
    input_length                    -> 1 int
    output_length                   -> 1 int
    layer_lengths                   -> num_layers ints
    activations                     -> num_layers + 1 ActivationEnum
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
    initialize_opencl(&vae->platform, &vae->device, &vae->context, &vae->queue, &vae->program, vae->kernels);

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
    vae->encoder.layer_activations =  malloc((vae->num_layers+1) * sizeof(ActivationEnum));
    vae->decoder.layer_activations =  malloc((vae->num_layers+1) * sizeof(ActivationEnum));
    fread(&vae->latent_space.length, sizeof(cl_int), 1, fptr);
    size = vae->latent_space.length * sizeof(cl_float);
    vae->latent_space.cl_sample_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);
    vae->latent_space.cl_distribution_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, 2*size, NULL, NULL);
    vae->latent_space.cl_epsilon_buffer = clCreateBuffer(vae->context, CL_MEM_READ_WRITE, size, NULL, NULL);

    fread(&vae->encoder.input_length, sizeof(cl_int), 1, fptr);
    fread(&vae->encoder.output_length, sizeof(cl_int), 1, fptr);
    fread(vae->encoder.layer_lengths, sizeof(cl_int), vae->num_layers, fptr);
    fread(vae->encoder.layer_activations, sizeof(ActivationEnum), vae->num_layers+1, fptr);
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
    fread(vae->decoder.layer_activations, sizeof(ActivationEnum), vae->num_layers+1, fptr);
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
    fwrite(&vae->latent_space.length, sizeof(cl_int), 1, fptr);

    fwrite(&vae->encoder.input_length, sizeof(cl_int), 1, fptr);
    fwrite(&vae->encoder.output_length, sizeof(cl_int), 1, fptr);
    fwrite(vae->encoder.layer_lengths, sizeof(cl_int), vae->num_layers, fptr);
    fwrite(vae->encoder.layer_activations, sizeof(ActivationEnum), vae->num_layers+1, fptr);
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
    fwrite(vae->decoder.layer_activations, sizeof(ActivationEnum), vae->num_layers+1, fptr);
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

int vae_get_latent_space_length(VAE* vae)
{
    return vae->latent_space.length;
}

// ==============================================================================================
// VAE implementation end
// ==============================================================================================
