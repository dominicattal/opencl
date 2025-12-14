#include "vae.h"
#include "mnist.h"
#include "fig.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void train_vae(void)
{
    MnistData* data;
    VAE* vae;
    const char* name;
    float* output;
    int img_width, img_height;
    int latent_space_length;
    int layer_lengths[2];
    int num_layers;
    int num_epochs;
    float eta, beta;
    ActivationEnum act;

    // params
    name = "pretrained/relu-512-256-10.vae";
    img_width = IMAGE_LENGTH;
    img_height = IMAGE_LENGTH;
    latent_space_length = 10;
    num_layers = 2;
    layer_lengths[0] = 512;
    layer_lengths[1] = 256;
    eta = 0.0005;
    beta = 2.0;
    num_epochs = 1;
    act = ACT_RELU;
    // ------

    data = mnist_load(MNIST_NUM_IMAGES);
    //vae = vae_create(img_width, img_height, latent_space_length, num_layers, layer_lengths, act);
    vae = vae_read(name);
    
    vae_train(vae, MNIST_NUM_IMAGES, data->buffers, eta, beta, num_epochs);

    write_image("images/in.png", img_width, img_height, data->buffers[1], 1);
    output = vae_feedforward(vae, data->buffers[1]);
    write_image("images/out.png", img_width, img_height, output, 5);

    free(output);
    vae_write(vae, name);
    vae_destroy(vae);
    mnist_destroy(data);
}

int main()
{
    train_vae();
    return 0;
}
