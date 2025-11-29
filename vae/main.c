#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <util.h>
#include <CL/cl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define IMAGE_LENGTH 28
#define NUM_PIXELS (IMAGE_LENGTH * IMAGE_LENGTH)

#define MNIST_TRAIN_IMAGES_COUNT    60000
#define MNIST_TRAIN_IMAGES_PATH     "../datasets/mnist/train-images.idx3-ubyte"
#define MNIST_TRAIN_LABELS_PATH     "../datasets/mnist/train-labels.idx1-ubyte"
#define MNIST_T10K_IMAGES_COUNT     10000
#define MNIST_T10K_IMAGES_PATH      "../datasets/mnist/t10k-images.idx3-ubyte"
#define MNIST_T10K_LABELS_PATH      "../datasets/mnist/t10k-images.idx1-ubyte"
#define MNIST_NUM_IMAGES            (MNIST_TRAIN_IMAGES_COUNT + MNIST_T10K_IMAGES_COUNT)

#define NUM_LAYERS  1

// ------------------------------------------------------------------

typedef struct {
    cl_float data[NUM_PIXELS];
} Image;

typedef enum {
    KERN_FORWARD,
    KERN_BACKWARD,
    KERN_SET_ERROR,
    NUM_KERNELS
} KernelEnum;

const char* kernel_names[NUM_KERNELS] = {
    "forward",
    "backward",
    "set_error"
};

typedef struct {
    cl_int num_rows;
    cl_int num_cols;
} MatrixSize;

typedef struct {
    cl_int          input_length;
    cl_int          output_length;
    cl_int*         layer_lengths;
    MatrixSize*     weight_sizes;
    cl_mem*         cl_buffers;
    cl_mem*         cl_weight_buffers;
} NeuralNet;

struct {

    // CL info
    cl_platform_id      platform;
    cl_device_id        device;
    cl_context          context;
    cl_command_queue    queue;
    cl_program          program;
    cl_kernel           kernels[NUM_KERNELS];

    // Model info
    cl_int              num_encoder_layers;
    cl_int              latent_space_length;
    cl_mem              cl_latent_space_buffer;
    cl_mem              cl_input_image_buffer;
    cl_mem              cl_output_image_buffer;
    cl_mem              cl_error_buffers[2];
    cl_float            learning_rate;
    NeuralNet           encoder;
    NeuralNet           decoder;

    // Image info
    int width, height, num_pixels;

} ctx;

// ------------------------------------------------------------------

static cl_int layer_lengths[NUM_LAYERS] = {2};

cl_float rand_float(void)
{
    return (cl_float)(rand() - (RAND_MAX>>1)) / RAND_MAX * 2; // [-1, 1]
}

void fill_buffer_random(cl_float* buffer, int num_rows, int num_cols)
{
    int i, j;
    for (i = 0; i < num_rows; i++)
        for (j = 0; j < num_cols; j++)
            buffer[i*num_cols+j] = rand_float();
}

void print_buf(cl_mem cl_buf, size_t size)
{
    cl_float* buf = malloc(size);
    clEnqueueReadBuffer(ctx.queue, cl_buf, CL_TRUE, 0, size, buf, 0, NULL, NULL);
    for (int i = 0; i*sizeof(cl_float) < size; i++)
        printf("%f ", buf[i]);
    puts("");
    puts("");
    free(buf);
}

void read_info(cl_platform_id platform)
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

cl_mem create_cl_buffer(size_t size)
{
    return clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, size, NULL, NULL);
}

void neuralnet_create(NeuralNet* net, cl_int input_length, cl_int output_length)
{
    cl_float* buf;
    size_t size;
    cl_int i, n, max_length;
    cl_int num_rows, num_cols;

    n = ctx.num_encoder_layers;
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
        net->cl_buffers[i] = create_cl_buffer(size);
    }

    num_rows = input_length+1; // add 1 for bias
    num_cols = net->layer_lengths[0];
    size = num_rows * num_cols * sizeof(cl_float);
    net->weight_sizes[0].num_rows = num_rows;
    net->weight_sizes[0].num_cols = num_cols;
    net->cl_weight_buffers[0] = create_cl_buffer(size);
    for (i = 1; i < n; i++) {
        num_rows = net->layer_lengths[i-1]+1;
        num_cols = net->layer_lengths[i];
        size = num_rows * num_cols * sizeof(cl_float);
        net->weight_sizes[i].num_rows = num_rows;
        net->weight_sizes[i].num_cols = num_cols;
        net->cl_weight_buffers[i] = create_cl_buffer(size);
    }
    num_rows = net->layer_lengths[n-1]+1;
    num_cols = output_length;
    size = num_rows * num_cols * sizeof(cl_float);
    net->weight_sizes[i].num_rows = num_rows;
    net->weight_sizes[i].num_cols = num_cols;
    net->cl_weight_buffers[n] = create_cl_buffer(size);

    // randomize weights
    for (i = 0; i <= n; i++) {
        num_rows = net->weight_sizes[i].num_rows;
        num_cols = net->weight_sizes[i].num_cols;
        size = num_rows * num_cols * sizeof(cl_float);
        fill_buffer_random(buf, num_rows, num_cols);
        clEnqueueWriteBuffer(ctx.queue, net->cl_weight_buffers[i], CL_TRUE, 0, size, buf, 0, NULL, NULL);
        clFinish(ctx.queue);
    }

    free(buf);
}

void neuralnet_destroy(NeuralNet* net)
{
    cl_int i;
    for (i = 0; i < ctx.num_encoder_layers; i++)
        clReleaseMemObject(net->cl_buffers[i]);
    for (i = 0; i <= ctx.num_encoder_layers; i++)
        clReleaseMemObject(net->cl_weight_buffers[i]);
    free(net->cl_weight_buffers);
    free(net->cl_buffers);
    free(net->weight_sizes);
}

void opencl_init(void)
{
    char* source;
    size_t size;
    int ret, n, i;

    n = NUM_LAYERS;

    clGetPlatformIDs(1, &ctx.platform, NULL);
    clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL);
    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, NULL);
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, NULL, NULL);
    source = read_file("vae.kern");
    ctx.program = clCreateProgramWithSource(ctx.context, 1, (const char**)&source, NULL, NULL);
    ret = clBuildProgram(ctx.program, 1, &ctx.device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        puts("clBuildProgram failed");
        char log[1<<16];
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 1<<16, log, NULL);
        puts(log);
        exit(1);
    }
    for (i = 0; i < NUM_KERNELS; i++)
        ctx.kernels[i] = clCreateKernel(ctx.program, kernel_names[i], NULL);

    read_info(ctx.platform);

    ctx.width = IMAGE_LENGTH;
    ctx.height = IMAGE_LENGTH;
    ctx.num_pixels = 2;
    ctx.num_encoder_layers = n;
    ctx.latent_space_length = 2;
    ctx.learning_rate = 0.5;

    size = (ctx.latent_space_length+1) * sizeof(cl_float);
    ctx.cl_latent_space_buffer = create_cl_buffer(size);
    size = (ctx.num_pixels+1) * sizeof(cl_float);
    ctx.cl_input_image_buffer = create_cl_buffer(size);
    ctx.cl_output_image_buffer = create_cl_buffer(size);
    ctx.cl_error_buffers[0] = create_cl_buffer(size);
    ctx.cl_error_buffers[1] = create_cl_buffer(size);

    ctx.encoder.layer_lengths = malloc(n * sizeof(cl_int));
    for (i = 0; i < n; i++)
        ctx.encoder.layer_lengths[i] = layer_lengths[i];
    ctx.decoder.layer_lengths = malloc(n * sizeof(cl_int));
    for (i = 0; i < n; i++)
        ctx.decoder.layer_lengths[n-i-1] = layer_lengths[i];

    neuralnet_create(&ctx.encoder, ctx.num_pixels, ctx.latent_space_length);
    neuralnet_create(&ctx.decoder, ctx.latent_space_length, ctx.num_pixels);
    
    free(source);
}

void opencl_cleanup(void)
{
    neuralnet_destroy(&ctx.encoder);
    neuralnet_destroy(&ctx.decoder);
    free(ctx.encoder.layer_lengths);
    free(ctx.decoder.layer_lengths);
    clReleaseMemObject(ctx.cl_latent_space_buffer);
    clReleaseMemObject(ctx.cl_input_image_buffer);
    clReleaseMemObject(ctx.cl_output_image_buffer);
    clReleaseMemObject(ctx.cl_error_buffers[0]);
    clReleaseMemObject(ctx.cl_error_buffers[1]);
    clReleaseProgram(ctx.program);
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
}

Image* mnist_load(void)
{
    unsigned char buf[NUM_PIXELS];
    unsigned char garbage[16];
    FILE* images_file;
    Image* images;
    int i, j;

    images = malloc(MNIST_NUM_IMAGES * sizeof(Image));

    images_file = fopen(MNIST_TRAIN_IMAGES_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    for (i = 0; i < MNIST_TRAIN_IMAGES_COUNT; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        for (j = 0; j < NUM_PIXELS; j++)
            images[i].data[j] = (cl_float)buf[j] / 255.0;
    }
    fclose(images_file);

    images_file = fopen(MNIST_T10K_IMAGES_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    for (i = MNIST_TRAIN_IMAGES_COUNT; i < MNIST_NUM_IMAGES; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        for (j = 0; j < NUM_PIXELS; j++)
            images[i].data[j] = (cl_float)buf[j] / 255.0;
    }
    fclose(images_file);

    return images;
}

void image_write(const char* filename, Image* image)
{
    unsigned char data[NUM_PIXELS];
    for (int i = 0; i < NUM_PIXELS; i++)
        data[i] = (unsigned char)roundf(image->data[i] * 255);
    stbi_write_png(filename, IMAGE_LENGTH, IMAGE_LENGTH, 1, data, 0);
}

void neuralnet_feedforward(NeuralNet* nn, cl_mem cl_buf_in, cl_int cl_buf_in_length, cl_mem cl_buf_out, cl_int cl_buf_out_length)
{
    cl_event* event_wait_list;
    cl_event event;
    cl_kernel kernel;
    size_t work_dim_size;
    int i, n, num_events, ret;
    n = ctx.num_encoder_layers;
    kernel = ctx.kernels[KERN_FORWARD];
    event_wait_list = malloc((n + 1) * sizeof(cl_event));
    num_events = 0;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[0]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &nn->cl_weight_buffers[0]);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &cl_buf_in_length);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &nn->layer_lengths[0]);
    work_dim_size = nn->layer_lengths[0];
    ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size, 
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
        ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size, 
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
    ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size, 
                           NULL, num_events, event_wait_list, &event);
    assert(ret == CL_SUCCESS);
    clFinish(ctx.queue);
    free(event_wait_list);
}

void neuralnet_backpropagate(NeuralNet* nn, cl_mem cl_buf_in, cl_int cl_buf_in_length, cl_mem cl_buf_out, cl_int cl_buf_out_length)
{
    cl_event* event_wait_list;
    cl_event event;
    cl_kernel kernel;
    size_t work_dim_size;
    cl_mem tmp;
    int i, j, n, num_events, ret;
    n = ctx.num_encoder_layers;
    kernel = ctx.kernels[KERN_BACKWARD];
    event_wait_list = malloc((n + 1) * sizeof(cl_event));
    num_events = 0;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[n-1]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buf_out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &ctx.cl_error_buffers[1]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &ctx.cl_error_buffers[0]);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[n]);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &nn->layer_lengths[n-1]);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &cl_buf_out_length);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &ctx.learning_rate);
    work_dim_size = nn->layer_lengths[n-1]+1;
    ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size,
                            NULL, 0, NULL, &event);
    clFinish(ctx.queue);
    assert(ret == CL_SUCCESS);
    event_wait_list[num_events++] = event;
    for (i = n-2; i >= 0; i--) {
        j = (n-i)%2;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn->cl_buffers[i]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[i+1]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &ctx.cl_error_buffers[j]);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &ctx.cl_error_buffers[1-j]);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[i+1]);
        clSetKernelArg(kernel, 5, sizeof(cl_int), &nn->layer_lengths[i]);
        clSetKernelArg(kernel, 6, sizeof(cl_int), &nn->layer_lengths[i+1]);
        clSetKernelArg(kernel, 7, sizeof(cl_int), &ctx.learning_rate);
        work_dim_size = nn->layer_lengths[i]+1;
        ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size,
                                NULL, num_events, event_wait_list, &event);
        assert(ret == CL_SUCCESS);
        event_wait_list[num_events++] = event;
    }
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nn->cl_buffers[0]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &ctx.cl_error_buffers[1-(n%2)]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &ctx.cl_error_buffers[n%2]);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &nn->cl_weight_buffers[0]);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &cl_buf_in_length);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &nn->layer_lengths[0]);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &ctx.learning_rate);
    work_dim_size = cl_buf_in_length+1;
    ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size,
                            NULL, num_events, event_wait_list, &event);
    assert(ret == CL_SUCCESS);
    clFinish(ctx.queue);
    free(event_wait_list);

    // ensure last error is in correct spot
    if (n % 2 == 0) {
        tmp = ctx.cl_error_buffers[0];
        ctx.cl_error_buffers[0] = ctx.cl_error_buffers[1];
        ctx.cl_error_buffers[1] = tmp;
    }
}

void vae_encode(void)
{
    neuralnet_feedforward(&ctx.encoder, ctx.cl_input_image_buffer, ctx.num_pixels,
            ctx.cl_latent_space_buffer, ctx.latent_space_length);
}

void vae_decode(void)
{
    neuralnet_feedforward(&ctx.decoder, ctx.cl_latent_space_buffer, ctx.latent_space_length, 
            ctx.cl_output_image_buffer, ctx.num_pixels);
}

void vae_feedforward(void)
{
    vae_encode();
    //vae_decode();
}

void vae_backpropagate(void)
{
    cl_kernel kernel;
    cl_int ret;
    size_t work_dim_size;
    kernel = ctx.kernels[KERN_SET_ERROR];
    cl_float target[] = {0.01, 0.99};
    cl_mem tmp = create_cl_buffer(sizeof(target));
    clEnqueueWriteBuffer(ctx.queue, tmp, CL_TRUE, 0, sizeof(target), target, 0, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &tmp);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ctx.cl_latent_space_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &ctx.cl_error_buffers[0]);
    //clSetKernelArg(kernel, 0, sizeof(cl_mem), &ctx.cl_input_image_buffer);
    //clSetKernelArg(kernel, 1, sizeof(cl_mem), &ctx.cl_latent_space_buffer);
    //clSetKernelArg(kernel, 2, sizeof(cl_mem), &ctx.cl_error_buffers[0]);
    work_dim_size = ctx.num_pixels;
    ret = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &work_dim_size,
                            NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    clFinish(ctx.queue);

    //neuralnet_backpropagate(&ctx.decoder, ctx.cl_latent_space_buffer, ctx.latent_space_length, 
    //        ctx.cl_output_image_buffer, ctx.num_pixels);
    neuralnet_backpropagate(&ctx.encoder, ctx.cl_input_image_buffer, ctx.num_pixels,
            ctx.cl_latent_space_buffer, ctx.latent_space_length);
}

Image* test_output(Image* image_in)
{
    size_t img_size = NUM_PIXELS * sizeof(cl_float);
    Image* image_out = malloc(sizeof(Image));
    clEnqueueWriteBuffer(ctx.queue, ctx.cl_input_image_buffer, CL_TRUE, 0, img_size, image_in->data, 0, NULL, NULL);
    vae_encode();
    vae_decode();
    clEnqueueReadBuffer(ctx.queue, ctx.cl_output_image_buffer, CL_TRUE, 0, img_size, image_out->data, 0, NULL, NULL);
    return image_out;
}

void print_all()
{
}

void vae_fit_step(Image* image)
{
    size_t img_size = ctx.num_pixels * sizeof(cl_float);
    clEnqueueWriteBuffer(ctx.queue, ctx.cl_input_image_buffer, CL_TRUE, 0, img_size, image->data, 0, NULL, NULL);
    vae_feedforward();
    vae_backpropagate();
}

void test(Image* image)
{
    image->data[0] = 0.05;
    image->data[1] = 0.1;
    cl_float W1[] = {0.15, 0.25, 0.2, 0.3, 0.35, 0.35};
    cl_float W2[] = {0.4, 0.5, 0.45, 0.55, 0.6, 0.6};
    cl_float buf[2];
    clEnqueueWriteBuffer(ctx.queue, ctx.cl_input_image_buffer, CL_TRUE, 0, 2*sizeof(cl_float), image->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx.queue, ctx.encoder.cl_weight_buffers[0], CL_TRUE, 0, sizeof(W1), W1, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx.queue, ctx.encoder.cl_weight_buffers[1], CL_TRUE, 0, sizeof(W2), W2, 0, NULL, NULL);
    for (int i = 0; i < 100; i++) {
        vae_feedforward();
        vae_backpropagate();
        clEnqueueReadBuffer(ctx.queue, ctx.cl_latent_space_buffer, CL_TRUE, 0, 8, buf, 0, NULL, NULL);
        printf("%f\n", 0.5 * (buf[0] - 0.01) * (buf[0] - 0.01) + 0.5 * (buf[1] - 0.99) * (buf[1] - 0.99));
    }
}

void vae_fit(int N, Image* images, int epochs, int batch_size)
{
    cl_float mse;
    cl_float data[NUM_PIXELS];
    int i, j, epoch;

    test(&images[0]);
    //for (epoch = 0; epoch < 10; epoch++) {
    //    for (i = 0; i < N; i++) {
    //        vae_fit_step(&images[0]);
    //        clEnqueueReadBuffer(ctx.queue, ctx.cl_latent_space_buffer, CL_TRUE, 0, 2 * sizeof(cl_float), data, 0, NULL, NULL);
    //        mse = 0;
    //        for (j = 0; j < 2; j++)
    //            mse += 0.5 * (images[i].data[j] - data[j]) * (images[i].data[j] - data[j]);
    //        printf("%d %f\n",i, mse);
    //    }
    //}
}

int main()
{
    Image* images = mnist_load();
    opencl_init();
    vae_fit(MNIST_NUM_IMAGES, images, 1, 1);
    opencl_cleanup();
    free(images);
}
