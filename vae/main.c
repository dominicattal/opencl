#include "vae.h"
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

int main()
{
    AutoEncoder* ae;
    VAE* vae;
    int img_width = IMAGE_LENGTH;
    int img_height = IMAGE_LENGTH;
    int latent_space_length = 10;
    int num_layers[] = {256, 256};
    int num_images = MNIST_NUM_IMAGES;
    int i;
    float* latent_space;
    float* input;
    float* output;
    float* heatmap;
    float** heatmaps;
    float y1,y2;
    MnistData data;
    const char* name = "pretrained/2.ae";

    data = mnist_load(num_images);

    //vae = vae_create(img_width, img_height, latent_space_length, sizeof(num_layers)/sizeof(int), num_layers);
    //vae = vae_read(name);
    //vae_train(vae, num_images, data.buffers, 0.1, 2);
    //vae_write(vae, name);
    //output = vae_feedforward(vae, data.buffers[2]);
    //write_image("images/in.png", data.buffers[2]);
    //write_image("images/out.png", output);
    //puts("x,y,label");
    //for (i = 0; i < num_images; i++) {
    //    output = vae_encode(vae, data.buffers[i]);
    //    printf("%f,%f,%d\n", output[0], output[1], data.labels[i]);
    //}
    //vae_destroy(vae);

    //ae = ae_create(img_width, img_height, latent_space_length, sizeof(num_layers)/sizeof(int), num_layers);
    ae = ae_read(name);

    //ae_train(ae, num_images, data.buffers, 0.1, 1);
    //ae_write(ae, name);

    input = data.buffers[2];
    output = ae_feedforward(ae, input);
    write_image("images/in.png", input);
    write_image("images/out.png", output);
    heatmaps = ae_create_heatmaps(ae, input);
    for (i = 0; i < latent_space_length; i++) {
        char buf[128];
        sprintf(buf, "images/heatmap%d.png", i);
        write_image(buf, heatmaps[i]);
    }
    //puts("x,y,label");
    //for (i = 0; i < num_images; i++) {
    //    latent_space = vae_encode(vae, data.buffers[i]);
    //    y1 = latent_space[0];
    //    y2 = latent_space[1];
    //    printf("%f,%f,%d\n", log(y1/(1-y1)), log(y2/(1-y2)), data.labels[i]);
    //    free(latent_space);
    //}

    //write_image("in.png", data[1]);
    //latent_space = vae_encode(vae, data[1]);
    //printf("%.20f\n%.20f\n", latent_space[0], latent_space[1]);
    //latent_space[0] += 0.1;
    //output = vae_decode(vae, latent_space);
    //write_image("out.png", output);
    //free(output);
    //free(latent_space);

    //ae_write(ae, name);
    //ae_destroy(ae);

    data_destroy(&data);
}
