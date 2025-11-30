#include "vae.h"
#include <stdlib.h>
#include <stdio.h>
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

float** mnist_load(void)
{
    unsigned char buf[NUM_PIXELS];
    unsigned char garbage[16];
    float** data;
    FILE* images_file;
    int i, j;

    data = malloc(MNIST_NUM_IMAGES * sizeof(float*));
    images_file = fopen(MNIST_TRAIN_IMAGES_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    for (i = 0; i < MNIST_TRAIN_IMAGES_COUNT; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        data[i] = malloc(NUM_PIXELS * sizeof(float));
        for (j = 0; j < NUM_PIXELS; j++)
            data[i][j] = (float)buf[j] / 255.0;
    }
    fclose(images_file);

    images_file = fopen(MNIST_T10K_IMAGES_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    for (i = MNIST_TRAIN_IMAGES_COUNT; i < MNIST_NUM_IMAGES; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        data[i] = malloc(NUM_PIXELS * sizeof(float));
        for (j = 0; j < NUM_PIXELS; j++)
            data[i][j] = (float)buf[j] / 255.0;
    }
    fclose(images_file);

    return data;
}

void write_image(const char* filename, float* data)
{
    unsigned char output[NUM_PIXELS];
    for (int i = 0; i < NUM_PIXELS; i++)
        output[i] = (unsigned char)roundf(data[i] * 255);
    stbi_write_png(filename, IMAGE_LENGTH, IMAGE_LENGTH, 1, output, 0);
}

int main()
{
    VAE* vae;
    int img_width = IMAGE_LENGTH;
    int img_height = IMAGE_LENGTH;
    int latent_space_length = 28;
    int num_layers[] = {200, 150};
    int i;
    float* output;
    float** data;

    //vae = vae_create(img_width, img_height, latent_space_length, sizeof(num_layers)/sizeof(int), num_layers);
    vae = vae_read("pretrained/1.vae");

    data = mnist_load();
    //vae_train(vae, MNIST_NUM_IMAGES, data, 0.1, 1);

    write_image("in.png", data[0]);
    output = vae_feedforward(vae, data[0]);
    write_image("out.png", output);
    free(output);

    //vae_write(vae, "pretrained/1.vae");
    vae_destroy(vae);

    for (i = 0; i < MNIST_NUM_IMAGES; i++)
        free(data[i]);
    free(data);
}
