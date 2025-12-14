#include "vae.h"
#include "mnist.h"
#include "fig.h"
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_SCALE 3

void interpolate_latent_space_vae(VAE* vae, float* in1, float* in2)
{
    Figure* fig;
    float* ls[2];
    float input_latent_space[2];
    float delta[2];
    float* output;
    int steps;
    int padding;
    int i, j;

    steps = 5;
    padding = 2;

    write_image("interp/vae/in1.png", IMAGE_LENGTH, IMAGE_LENGTH, in1, DEFAULT_SCALE);
    write_image("interp/vae/in2.png", IMAGE_LENGTH, IMAGE_LENGTH, in2, DEFAULT_SCALE);

    ls[0] = vae_encode(vae, in1);
    ls[1] = vae_encode(vae, in2);
    printf("in1: %f %f\n", ls[0][0], ls[0][1]);
    printf("in2: %f %f\n", ls[1][0], ls[1][1]);
    delta[0] = (ls[1][0] - ls[0][0]) / steps;
    delta[1] = (ls[1][1] - ls[0][1]) / steps;

    output = vae_decode(vae, ls[0]);
    write_image("interp/vae/out1.png", IMAGE_LENGTH, IMAGE_LENGTH, output, DEFAULT_SCALE);
    output = vae_decode(vae, ls[1]);
    write_image("interp/vae/out2.png", IMAGE_LENGTH, IMAGE_LENGTH, output, DEFAULT_SCALE);

    fig = figure_create(IMAGE_LENGTH, IMAGE_LENGTH, steps+1, steps+1, padding);

    for (i = 0; i <= steps; i++) {
        for (j = 0; j <= steps; j++) {
            input_latent_space[0] = ls[0][0] + i * delta[0];
            input_latent_space[1] = ls[0][1] + j * delta[1];
            output = vae_decode(vae, input_latent_space);
            figure_add(fig, i, j, output);
            free(output);
        }
    }

    figure_write(fig, "interp/vae/output.png");
    figure_destroy(fig);
    free(ls[0]);
    free(ls[1]);
}

void interpolate_latent_space_ae(AutoEncoder* ae, float* in1, float* in2)
{
    Figure* fig;
    float* ls[2];
    float input_latent_space[2];
    float delta[2];
    float* output;
    int steps;
    int padding;
    int i, j;

    steps = 5;
    padding = 2;

    write_image("interp/ae/in1.png", IMAGE_LENGTH, IMAGE_LENGTH, in1, DEFAULT_SCALE);
    write_image("interp/ae/in2.png", IMAGE_LENGTH, IMAGE_LENGTH, in2, DEFAULT_SCALE);

    ls[0] = ae_encode(ae, in1);
    ls[1] = ae_encode(ae, in2);
    printf("in1: %f %f\n", ls[0][0], ls[0][1]);
    printf("in2: %f %f\n", ls[1][0], ls[1][1]);
    delta[0] = (ls[1][0] - ls[0][0]) / steps;
    delta[1] = (ls[1][1] - ls[0][1]) / steps;

    output = ae_decode(ae, ls[0]);
    write_image("interp/ae/out1.png", IMAGE_LENGTH, IMAGE_LENGTH, output, DEFAULT_SCALE);
    output = ae_decode(ae, ls[1]);
    write_image("interp/ae/out2.png", IMAGE_LENGTH, IMAGE_LENGTH, output, DEFAULT_SCALE);

    fig = figure_create(IMAGE_LENGTH, IMAGE_LENGTH, steps+1, steps+1, padding);

    for (i = 0; i <= steps; i++) {
        for (j = 0; j <= steps; j++) {
            input_latent_space[0] = ls[0][0] + i * delta[0];
            input_latent_space[1] = ls[0][1] + j * delta[1];
            output = ae_decode(ae, input_latent_space);
            figure_add(fig, i, j, output);
            free(output);
        }
    }
    figure_write(fig, "interp/ae/output.png");
    figure_destroy(fig);
    free(ls[0]);
    free(ls[1]);
}

int main()
{
    MnistData* data;
    AutoEncoder* ae;
    VAE* vae;
    float* in1;
    float* in2;
    data = mnist_load(MNIST_NUM_IMAGES);
    in1 = data->buffers[1];
    in2 = data->buffers[3];
    ae = ae_read("pretrained/sigmoid-256-256-2-2.ae");
    vae = vae_read("pretrained/sigmoid-256-256-2.vae");
    interpolate_latent_space_ae(ae, in1, in2);
    interpolate_latent_space_vae(vae, in1, in2);
    vae_destroy(vae);
    ae_destroy(ae);
    mnist_destroy(data);
}
