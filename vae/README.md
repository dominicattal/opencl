# Variational Autoencoder

[What is a Variational Autoencoder](https://www.ibm.com/think/topics/variational-autoencoder) 

Build using `make`

```
interp   - create interpolation between two images
csv      - write latent space to csv
traverse - traverse across latent dimensions
heatmap  - create heatmap of output wrt latent dimensions
train    - train a model
```
## Training
<p>
  This program will train a model specified by the parameters in <code>train.c</code>. The customizable parameters were
  <pre>
    latent_space_length -> how many dimensions the latent space should be.
    num_layers          -> the number of layers the separate the input from the latent space and the latent space from the output.
    layer_lengths       -> integer array containing the size of each hidden layer in the neural network. this array represents the input to the latent space and is mirrored from the latent space to the output.
    eta                 -> learning rate for model training
    beta                -> coefficient for learning the statistical model (vae only)
    num_epochs          -> number of times to train on dataset
    act                 -> the activiation function to use. the two that were implemented were ACT_SIGMOID and ACT_RELU.
  </pre>
  The program will write the data for the model to the path specified in the first command-line argument to the program. The model can then be read in the other programs
  below.

  From my experience, it takes a couple of tries of tweaking input parameters to train a model that performs well. Some common pitfalls I've run into are
  - vanishing gradients, typically for sigmoid activation functions, that stop training
  - exploding gradients, when eta or beta was too high
  - model collapse, where the model learns a model that produces the same image for each latent space

  I've trained good models with both the sigmoid and relu activation functions though, and you can read them from the <code>pretrained</code> folder. Some of the programs take
  from these models by default, but you can change them with any model you train.
</p>

## Interpolation
 
<p>
  This program will take some handwritten digits as input and create a latent space traversal image as an output. It does this by translating the input images
  in its latent space and linearly interpolating each dimension of the latent space. This program is meant for models with a latent space size of 2. Here is a sample output.
</p>

<table align="center">
  <h3 align="center">Input</h3>
  <tr>
    <td align="center">
<img width="84" height="84" alt="in1" src="https://github.com/user-attachments/assets/c17c655b-6705-4217-a287-ca6d482fd615" />
    </td>
    <td align="center" valign="middle">
      <strong>→</strong>
    </td>
    <td align="center">
<img width="84" height="84" alt="in2" src="https://github.com/user-attachments/assets/cfafb81d-115d-4acd-98df-fd85fc8adedd" />
    </td>
  </tr>
</table>


<table align="center">
  <h3 align="center">Output</h3>
  <tr>
    <td align="center">
      <img width="84" height="84" alt="out1" src="https://github.com/user-attachments/assets/59984800-9e47-4cdd-9ad8-15c5e4c9f411" />
    </td>
    <td align="center" valign="middle">
      <strong>→</strong>
    </td>
    <td align="center">
      <img width="84" height="84" alt="out2" src="https://github.com/user-attachments/assets/9daa3692-a309-49fd-b695-6f0ff3acae97" />
    </td>
  </tr>
</table>

<div align="center">
  <h3>Full Interpolation</h3>
  <img align="center" width="356" height="356" alt="output" src="https://github.com/user-attachments/assets/d6004faa-2e51-4c24-b72f-6319cd7b3a73" />
</div>

## CSV

<p>
This program will translate each image from the MNIST dataset into a latent space representation formatted as a csv file. This can be used with 
<code>plot_latent_space.py</code> to visualize the performance of a model.
</p>

<div align="center">
  <img width="530" height="510" alt="graph" src="https://github.com/user-attachments/assets/2b482011-cb05-4af7-a46a-1ae0c1aa19ce" />
  <p>This shows how the model creates "regions" in latent space where certain categories of images are likely to appear.</p>
</div>


## Traverse

<p>
  This program will take some handwritten digits as input and create a latent space traversal image as an output. Unlike the Interpolation program, it is made to work
  for any latent space size. It uses the <em>ceteris paribus</em> idea to highlight how each latent space dimension affects the output. It also writes each image separately 
  to a folder so you can manipulate them with ffmpeg.
</p>

<div align="center">
  <img width="626" height="270" alt="Untitled Diagram" src="https://github.com/user-attachments/assets/54a286a4-4f44-40f2-9c85-cbd07b0b932b" />
  <p width="500">Here, the model takes in an image, compresses it to a latent space, then rebuilds it from the latent space. It would be useful to know how each dimension
  affects the output.</p>
</div>

## Heatmap
<p>
  This program will create a heatmap of each of the latent space dimensions that explains which pixels are most affected by that dimension.
</p>

<div align="center">
<img width="762" height="320" alt="Untitled Diagram(1)" src="https://github.com/user-attachments/assets/a364355d-b56e-4f70-8e8b-ca61f6c5a94f" />
<p width="500">Heatmap for the same image from the traverse program. It shows how the 3rd (top middle) dimension is outlining the shape of the final image, which suggests
this dimension is responsible for the thickness of the output.</p>
</div>
