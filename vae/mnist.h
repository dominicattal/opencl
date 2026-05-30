#ifndef MNIST_H
#define MNIST_H

#define IMAGE_LENGTH 28
#define NUM_PIXELS (IMAGE_LENGTH * IMAGE_LENGTH)
#define MNIST_TRAIN_IMAGES_COUNT    60000
#define MNIST_TRAIN_IMAGES_PATH     "../datasets/mnist/train-images.idx3-ubyte"
#define MNIST_TRAIN_LABELS_PATH     "../datasets/mnist/train-labels.idx1-ubyte"
#define MNIST_T10K_IMAGES_COUNT     10000
#define MNIST_T10K_IMAGES_PATH      "../datasets/mnist/t10k-images.idx3-ubyte"
#define MNIST_T10K_LABELS_PATH      "../datasets/mnist/t10k-labels.idx1-ubyte"
#define MNIST_NUM_IMAGES            (MNIST_TRAIN_IMAGES_COUNT + MNIST_T10K_IMAGES_COUNT)

typedef struct {
    float** buffers;
    unsigned char* labels;
    int num_images;
} MnistData;

// loads num_images images from the mnist data, max MNIST_NUM_IMAGES
MnistData*  mnist_load(int num_images);
void        mnist_destroy(MnistData* data);

#endif
