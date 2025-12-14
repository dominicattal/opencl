#include "fig.h"
#include "vae.h"
#include "mnist.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#define DEFAULT_SCALE 3

void makedir(const char* filename)
{
#ifdef __WIN32
    mkdir(filename);
#else
    mkdir(filename, 0777);
#endif
}

void traverse_vae(VAE* vae, float* in)
{
    Figure* fig;
    float* ls;
    float* new_ls;
    float* output;
    float step;
    int num_steps;
    int padding;
    int i, j;
    int latent_space_length;
    char buf[128];

    num_steps = 100;
    padding = 3;
    step = 0.1;
    latent_space_length = vae_get_latent_space_length(vae);

    write_image("traverse/vae/in.png", IMAGE_LENGTH, IMAGE_LENGTH, in, 1);
    output = vae_feedforward(vae, in);
    write_image("traverse/vae/out.png", IMAGE_LENGTH, IMAGE_LENGTH, output, 1);
    free(output);

    fig = figure_create(IMAGE_LENGTH, IMAGE_LENGTH, latent_space_length, 2*num_steps+1, padding);

    new_ls = malloc(latent_space_length * sizeof(float));
    ls = vae_encode(vae, in); 
    for (i = 0; i < latent_space_length; i++)
        printf("%f ", ls[i]);
    puts("");
    makedir("traverse/vae/imgs");
    memcpy(new_ls, ls, latent_space_length * sizeof(float));
    for (i = 0; i < latent_space_length; i++) {
        sprintf(buf, "traverse/vae/imgs/%d", i+1);
        makedir(buf);
        for (j = 0; j < 2*num_steps+1; j++) {
            new_ls[i] = ls[i] + step * (j-num_steps);
            output = vae_decode(vae, new_ls);
            figure_add(fig, i, j, output);
            sprintf(buf, "traverse/vae/imgs/%d/%d.png", i+1, j);
            write_image(buf, IMAGE_LENGTH, IMAGE_LENGTH, output, DEFAULT_SCALE);
            free(output);
        }
        new_ls[i] = ls[i];
    }
    figure_write(fig, "traverse/vae/traversal.png");
    figure_destroy(fig);
    free(ls);
    free(new_ls);
}

void traverse_ae(AutoEncoder* ae, float* in)
{
    Figure* fig;
    float* ls;
    float* new_ls;
    float* output;
    float step;
    int num_steps;
    int padding;
    int i, j;
    int latent_space_length;
    char buf[128];

    num_steps = 100;
    padding = 3;
    step = 0.1;
    latent_space_length = ae_get_latent_space_length(ae);

    write_image("traverse/ae/in.png", IMAGE_LENGTH, IMAGE_LENGTH, in, 1);
    output = ae_feedforward(ae, in);
    write_image("traverse/ae/out.png", IMAGE_LENGTH, IMAGE_LENGTH, output, 1);
    free(output);

    fig = figure_create(IMAGE_LENGTH, IMAGE_LENGTH, latent_space_length, 2*num_steps+1, padding);

    new_ls = malloc(latent_space_length * sizeof(float));
    ls = ae_encode(ae, in); 
    for (i = 0; i < latent_space_length; i++)
        printf("%f ", ls[i]);
    puts("");
    makedir("traverse/ae/imgs");
    memcpy(new_ls, ls, latent_space_length * sizeof(float));
    for (i = 0; i < latent_space_length; i++) {
        sprintf(buf, "traverse/ae/imgs/%d", i+1);
        makedir(buf);
        for (j = 0; j < 2*num_steps+1; j++) {
            new_ls[i] = ls[i] + step * (j-num_steps);
            output = ae_decode(ae, new_ls);
            figure_add(fig, i, j, output);
            sprintf(buf, "traverse/ae/imgs/%d/%d.png", i+1, j);
            write_image(buf, IMAGE_LENGTH, IMAGE_LENGTH, output, DEFAULT_SCALE);
            free(output);
        }
        new_ls[i] = ls[i];
    }
    figure_write(fig, "traverse/ae/traversal.png");
    figure_destroy(fig);
    free(ls);
    free(new_ls);
}

int main()
{
    MnistData* data;
    AutoEncoder* ae;
    VAE* vae;
    float* in;
    data = mnist_load(MNIST_NUM_IMAGES);
    in = data->buffers[1];
    ae = ae_read("pretrained/sigmoid-256-256-2-2.ae");
    vae = vae_read("pretrained/relu-512-256-10.vae");
    traverse_vae(vae, in);
    //traverse_ae(ae, in);
    vae_destroy(vae);
    ae_destroy(ae);
    mnist_destroy(data);
}
