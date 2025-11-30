#ifndef VAE_H
#define VAE_H

typedef struct VAE VAE;

VAE*    vae_create(int img_width, int img_height, int latent_space_length, int num_layers, int* layer_lengths);
void    vae_train(VAE* vae, int num_images, float** image_data, float learning_rate, int epochs);
float*  vae_feedforward(VAE* vae, float* data);
float*  vae_encode(VAE* vae, float* data);
float*  vae_decode(VAE* vae, float* data);
float*  vae_create_heatmap(VAE* vae, float* in_data, float* out_data);
void    vae_write(VAE* vae, const char* filename);
VAE*    vae_read(const char* filename);
void    vae_destroy(VAE* vae);

#endif
