#include "mnist.h"
#include "fig.h"
#include "vae.h"
#include <stdio.h>

void latent_space_csv_vae(VAE* vae, MnistData* data)
{
    float* output;
    FILE* fptr;
    fptr = fopen("csv/latent_space_vae.csv", "wb");
    fprintf(fptr, "x,y,label\n");
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        output = vae_encode(vae, data->buffers[i]);
        fprintf(fptr, "%f,%f,%d\n", output[0], output[1], data->labels[i]);
    }
    fclose(fptr);
}

void latent_space_csv_ae(AutoEncoder* ae, MnistData* data)
{
    float* output;
    FILE* fptr;
    fptr = fopen("csv/latent_space_ae.csv", "wb");
    fprintf(fptr, "x,y,label\n");
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        output = ae_encode(ae, data->buffers[i]);
        fprintf(fptr, "%f,%f,%d\n", output[0], output[1], data->labels[i]);
    }
    fclose(fptr);
}

int main()
{
    MnistData* data;
    AutoEncoder* ae;
    VAE* vae;
    data = mnist_load(MNIST_NUM_IMAGES);
    ae = ae_read("pretrained/sigmoid-256-256-2-2.ae");
    vae = vae_read("pretrained/relu-512-256-2.vae");
    //latent_space_csv_ae(ae, data);
    latent_space_csv_vae(vae, data);
    vae_destroy(vae);
    ae_destroy(ae);
    mnist_destroy(data);
}
