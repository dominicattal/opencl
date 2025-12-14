#include "fig.h"
#include "mnist.h"
#include "vae.h"
#include <stdio.h>
#include <stdlib.h>

void heatmap_vae(VAE* vae, float* in)
{
    Figure* fig;
    float** heatmaps;
    float* output;
    int padding;
    int latent_space_length;
    int i;

    padding = 3;
    latent_space_length = vae_get_latent_space_length(vae);

    fig = figure_create(IMAGE_LENGTH, IMAGE_LENGTH, 1, latent_space_length+2, padding);

    output = vae_feedforward(vae, in);
    heatmaps = vae_create_heatmaps(vae, output);
    figure_add(fig, 0, 0, in);
    figure_add(fig, 0, 1, output);
    free(output);
    output = vae_encode(vae, in);
    for (i = 0; i < latent_space_length; i++) {
        figure_add(fig, 0, i+2, heatmaps[i]);
        free(heatmaps[i]);
    }
    figure_write(fig, "heatmap/vae.png");
    figure_destroy(fig);
    free(output);
    free(heatmaps);
}

void heatmap_ae(AutoEncoder* ae, float* in)
{
    Figure* fig;
    float** heatmaps;
    float* output;
    int padding;
    int latent_space_length;
    int i;

    padding = 3;
    latent_space_length = ae_get_latent_space_length(ae);

    fig = figure_create(IMAGE_LENGTH, IMAGE_LENGTH, 1, latent_space_length+2, padding);

    output = ae_feedforward(ae, in);
    heatmaps = ae_create_heatmaps(ae, output);
    figure_add(fig, 0, 0, in);
    figure_add(fig, 0, 1, output);
    free(output);
    output = ae_encode(ae, in);
    for (i = 0; i < latent_space_length; i++) {
        figure_add(fig, 0, i+2, heatmaps[i]);
        free(heatmaps[i]);
    }
    figure_write(fig, "heatmap/ae.png");
    figure_destroy(fig);
    free(output);
    free(heatmaps);
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
    vae = vae_read("pretrained/400-200-10.vae");
    heatmap_vae(vae, in);
    heatmap_ae(ae, in);
    vae_destroy(vae);
    ae_destroy(ae);
    mnist_destroy(data);
}
