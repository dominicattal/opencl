#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

MnistData* mnist_load(int num_images)
{
    unsigned char buf[NUM_PIXELS];
    unsigned char garbage[16];
    MnistData* data;
    FILE* images_file;
    FILE* labels_file;
    int i, j;

    assert(num_images <= MNIST_NUM_IMAGES);

    data = malloc(sizeof(MnistData));
    data->buffers = malloc(num_images * sizeof(float*));
    data->labels = malloc(num_images * sizeof(int));
    data->num_images = 0;
    images_file = fopen(MNIST_TRAIN_IMAGES_PATH, "rb");
    labels_file = fopen(MNIST_TRAIN_LABELS_PATH, "rb");
    assert(images_file != NULL);
    fread(garbage, sizeof(unsigned char), 16, images_file);
    fread(garbage, sizeof(unsigned char), 8, labels_file);
    for (i = 0; i < num_images && i < MNIST_TRAIN_IMAGES_COUNT; i++) {
        fread(buf, sizeof(unsigned char), NUM_PIXELS, images_file);
        data->buffers[i] = malloc(NUM_PIXELS * sizeof(float));
        for (j = 0; j < NUM_PIXELS; j++)
            data->buffers[i][j] = (float)buf[j] / 255.0;
        fread(&data->labels[i], sizeof(unsigned char), 1, labels_file);
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
        data->buffers[i] = malloc(NUM_PIXELS * sizeof(float));
        for (j = 0; j < NUM_PIXELS; j++)
            data->buffers[i][j] = (float)buf[j] / 255.0;
        fread(&data->labels[i], sizeof(unsigned char), 1, labels_file);
    }
    fclose(images_file);
    fclose(labels_file);

    return data;
}

void mnist_destroy(MnistData* data)
{
    for (int i = 0; i < data->num_images; i++)
        free(data->buffers[i]);
    free(data->buffers);
    free(data->labels);
    free(data);
}
