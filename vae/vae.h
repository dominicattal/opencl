#ifndef VAE_H
#define VAE_H

typedef enum {
    ACT_NONE,
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
    NUM_ACTIVATIONS
} ActivationEnum;

typedef struct AutoEncoder AutoEncoder;
typedef struct VAE VAE;

AutoEncoder*    ae_create(int img_width, int img_height, int latent_space_length, int num_layers, int* layer_lengths, ActivationEnum activation);
void            ae_train(AutoEncoder* ae, int num_images, float** image_data, float learning_rate, int epochs);
float*          ae_feedforward(AutoEncoder* ae, float* data);
float*          ae_encode(AutoEncoder* ae, float* data);
float*          ae_decode(AutoEncoder* ae, float* data);
float*          ae_create_heatmap(AutoEncoder* ae, float* input, int latent_space_idx);
float**         ae_create_heatmaps(AutoEncoder* ae, float* input);
void            ae_write(AutoEncoder* ae, const char* filename);
AutoEncoder*    ae_read(const char* filename);
void            ae_destroy(AutoEncoder* ae);
int             ae_get_latent_space_length(AutoEncoder* ae);

VAE*            vae_create(int img_width, int img_height, int latent_space_length, int num_layers, int* layer_lengths, ActivationEnum activation);
void            vae_seed(VAE* vae, unsigned long long seed);
void            vae_train(VAE* vae, int num_images, float** image_data, float learning_rate, float beta, int epochs);
float*          vae_feedforward(VAE* vae, float* data);
float*          vae_encode(VAE* vae, float* data);
float*          vae_decode(VAE* vae, float* data);
float**         vae_create_heatmaps(VAE* vae, float* data);
void            vae_write(VAE* vae, const char* filename);
VAE*            vae_read(const char* filename);
void            vae_destroy(VAE* vae);
int             vae_get_latent_space_length(VAE* vae);

#endif
