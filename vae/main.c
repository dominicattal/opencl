#include "vae.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
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

typedef struct {
    float** buffers;
    unsigned char* labels;
    int num_images;
} MnistData;

MnistData mnist_load(int num_images)
{
    unsigned char buf[NUM_PIXELS];
    unsigned char garbage[16];
    MnistData data;
    FILE* images_file;
    FILE* labels_file;
    int i, j;

    data.buffers = malloc(num_images * sizeof(float*));
    data.labels = malloc(num_images * sizeof(int));
    data.num_images = 0;
    images_file = fopen(MNIST_TRAIN_IMAGES_PATH, "rb");
    labels_file = fopen(MNIST_TRAIN_LABELS_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    fread(garbage, sizeof(unsigned char), 8, labels_file);
    for (i = 0; i < num_images && i < MNIST_TRAIN_IMAGES_COUNT; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        data.buffers[i] = malloc(NUM_PIXELS * sizeof(float));
        for (j = 0; j < NUM_PIXELS; j++)
            data.buffers[i][j] = (float)buf[j] / 255.0;
        fread(&data.labels[i], sizeof(unsigned char), 1, labels_file);
    }
    fclose(images_file);
    fclose(labels_file);

    images_file = fopen(MNIST_T10K_IMAGES_PATH, "rb");
    labels_file = fopen(MNIST_T10K_LABELS_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    fread(garbage, sizeof(unsigned char), 8, labels_file);
    for (i = MNIST_TRAIN_IMAGES_COUNT; i < num_images && i < MNIST_NUM_IMAGES; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        data.buffers[i] = malloc(NUM_PIXELS * sizeof(float));
        for (j = 0; j < NUM_PIXELS; j++)
            data.buffers[i][j] = (float)buf[j] / 255.0;
        fread(&data.labels[i], sizeof(unsigned char), 1, labels_file);
    }
    fclose(images_file);
    fclose(labels_file);

    return data;
}

void write_image(const char* filename, float* data)
{
    unsigned char output[NUM_PIXELS];
    for (int i = 0; i < NUM_PIXELS; i++)
        output[i] = (unsigned char)roundf(data[i] * 255);
    stbi_write_png(filename, IMAGE_LENGTH, IMAGE_LENGTH, 1, output, 0);
}

void data_destroy(MnistData* data)
{
    for (int i = 0; i < data->num_images; i++)
        free(data->buffers[i]);
    free(data->buffers);
    free(data->labels);
}

void interpolate_latent_space(VAE* vae, float* in1, float* in2)
{
    float* ls[2];
    float input_latent_space[2];
    float delta[2];
    unsigned char* img;
    float* output;
    int steps = 5;
    int padding = 5;
    int width_with_padding;
    int num_pixels;
    int i, j;

    write_image("interp/in1.png", in1);
    write_image("interp/in2.png", in2);
    vae_seed(vae, 1000);

    width_with_padding = IMAGE_LENGTH * (steps+1) + padding * steps;
    num_pixels = width_with_padding * IMAGE_LENGTH * (steps+1) 
                   + width_with_padding * padding * steps;
    ls[0] = vae_encode(vae, in1);
    ls[1] = vae_encode(vae, in2);
    printf("in1: %f %f\n", ls[0][0], ls[0][1]);
    printf("in2: %f %f\n", ls[1][0], ls[1][1]);
    delta[0] = (ls[1][0] - ls[0][0]) / steps;
    delta[1] = (ls[1][1] - ls[0][1]) / steps;
    img = malloc(num_pixels * sizeof(unsigned char));
    for (i = 0; i < num_pixels; i++)
        img[i] = 255;

    output = vae_decode(vae, ls[0]);
    write_image("interp/out1.png", output);
    output = vae_decode(vae, ls[1]);
    write_image("interp/out2.png", output);

    void img_write(float* data, int row, int col)
    {
        int x = col * (IMAGE_LENGTH + padding);
        int y = row * (IMAGE_LENGTH + padding);
        for (int dy = 0; dy < IMAGE_LENGTH; dy++)
            for (int dx = 0; dx < IMAGE_LENGTH; dx++)
                img[(y+dy)*width_with_padding+(x+dx)] = (unsigned char)roundf(data[dy*IMAGE_LENGTH+dx] * 255);
    }

    for (i = 0; i <= steps; i++) {
        for (j = 0; j <= steps; j++) {
            input_latent_space[0] = ls[0][0] + i * delta[0];
            input_latent_space[1] = ls[0][1] + j * delta[1];
            output = vae_decode(vae, input_latent_space);
            img_write(output, i, j);
            free(output);
        }
    }
    stbi_write_png("interp/output.png", width_with_padding, width_with_padding, 1, img, 0);
    free(ls[0]);
    free(ls[1]);
    free(img);
}

void latent_space_csv(VAE* vae, MnistData* data)
{
    float* output;
    FILE* fptr;
    fptr = fopen("latent_space_vae.csv", "wb");
    fprintf(fptr, "x,y,label\n");
    vae_seed(vae, 1000);
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        output = vae_encode(vae, data->buffers[i]);
        fprintf(fptr, "%f,%f,%d\n", output[0], output[1], data->labels[i]);
    }
    fclose(fptr);
}

void interpolate_latent_space10(VAE* vae, float* in)
{
    float* ls;
    float new_ls[10];
    float* output;
    unsigned char* img;
    float step = 0.1;
    int num_steps = 100;
    int padding = 3;
    int width_with_padding;
    int height_with_padding;
    int num_pixels;
    int i, j;

    write_image("interp10/in.png", in);
    vae_seed(vae, 1000);

    width_with_padding = IMAGE_LENGTH * (2*num_steps+1) + padding * (2*num_steps);
    height_with_padding = 10 * IMAGE_LENGTH + 9 * padding;
    num_pixels = width_with_padding * height_with_padding;
    img = malloc(num_pixels * sizeof(unsigned char));
    for (i = 0; i < num_pixels; i++)
        img[i] = 255;
    output = malloc(IMAGE_LENGTH * IMAGE_LENGTH * sizeof(float));
    for (i = 0; i < IMAGE_LENGTH * IMAGE_LENGTH; i++)
        output[i] = 0.0f;

    void img_write(float* data, int row, int col)
    {
        int x = col * (IMAGE_LENGTH + padding);
        int y = row * (IMAGE_LENGTH + padding);
        for (int dy = 0; dy < IMAGE_LENGTH; dy++)
            for (int dx = 0; dx < IMAGE_LENGTH; dx++)
                img[(y+dy)*width_with_padding+(x+dx)] = (unsigned char)roundf(data[dy*IMAGE_LENGTH+dx] * 255);
    }

    ls = vae_encode(vae, in); 
    memcpy(new_ls, ls, 10 * sizeof(float));
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 2*num_steps+1; j++) {
            new_ls[i] = ls[i] + step * (j-num_steps);
            output = vae_decode(vae, new_ls);
            img_write(output, i, j);
            free(output);
        }
        new_ls[i] = ls[i];
    }
    stbi_write_png("interp10/out.png", width_with_padding, height_with_padding, 1, img, 0);
    free(img);
}

void create_heatmap10(VAE* vae, float* in)
{
    float** heatmaps;
    float* output_img;
    float* output;
    unsigned char* img;
    int padding = 3;
    int width_with_padding;
    int num_pixels;
    int i;

    width_with_padding = IMAGE_LENGTH * 12 + padding * 11;
    num_pixels = width_with_padding * IMAGE_LENGTH;
    img = malloc(num_pixels * sizeof(unsigned char));
    for (i = 0; i < num_pixels; i++)
        img[i] = 255;

    void img_write(float* data, int col)
    {
        int x = col * (IMAGE_LENGTH + padding);
        for (int dy = 0; dy < IMAGE_LENGTH; dy++)
            for (int dx = 0; dx < IMAGE_LENGTH; dx++)
                img[dy*width_with_padding+(x+dx)] = (unsigned char)roundf(data[dy*IMAGE_LENGTH+dx] * 255);
    }

    vae_seed(vae, 1005);
    output_img = vae_feedforward(vae, in);
    heatmaps = vae_create_heatmaps(vae, output_img);
    output = vae_encode(vae, in);
    img_write(in, 0);
    img_write(output_img, 1);
    for (i = 0; i < 10; i++) {
        img_write(heatmaps[i], i+2);
        free(heatmaps[i]);
    }
    stbi_write_png("heatmaps10/heatmaps.png", width_with_padding, IMAGE_LENGTH, 1, img, 0);
    free(output);
    free(heatmaps);
    free(img);
}

void train(MnistData* data)
{
    int layer_lengths[] = {256, 64};
    int num_images = MNIST_NUM_IMAGES;
    int latent_space_length = 10;
    VAE* vae;
    float* input;
    float* output;
    vae = vae_create(IMAGE_LENGTH, IMAGE_LENGTH, latent_space_length, 2, layer_lengths);
    vae = vae_read("pretrained/test10.vae");
    vae_train(vae, MNIST_NUM_IMAGES, data->buffers, 0.001, 1);
    vae_write(vae, "pretrained/test10.vae");
    input = data->buffers[1];
    output = vae_feedforward(vae, input);
    write_image("images/in.png", input);
    write_image("images/out.png", output);
    vae_destroy(vae);
    free(output);
}

void test2(MnistData* data)
{
    VAE* vae;
    float* in1;
    float* in2;
    const char* name;
    name = "pretrained/relu-002-256-64-2.vae";
    //name = "pretrained/sigmoid-256-256-2.vae";
    vae = vae_read(name);
    in1 = data->buffers[1];
    in2 = data->buffers[3];
    interpolate_latent_space(vae, in1, in2);
    latent_space_csv(vae, data);

    vae_destroy(vae);
}

void test10(MnistData* data)
{
    VAE* vae;
    float* input;
    float* output;
    const char* name;
    //name = "pretrained/relu-256-64-10.vae";
    name = "pretrained/400-200-10.vae";
    vae = vae_read(name);
    input = data->buffers[3];
    interpolate_latent_space10(vae, input);
    create_heatmap10(vae, input);
    output = vae_feedforward(vae, input);
    write_image("images/in.png", input);
    write_image("images/out.png", output);
    vae_destroy(vae);
}

int main()
{
    MnistData data;

    srand(time(NULL));
    data = mnist_load(MNIST_NUM_IMAGES);
    train(&data);
    //test10(&data);
    //test2(&data);
    data_destroy(&data);
}
