import pandas as pd
import matplotlib.pyplot as plt
def create_plot(filename, output_filename):
    df = pd.read_csv(filename)
    plt.figure(figsize=(20,20))
    digits = range(10)
    for dig in digits:
      rows = df[df["label"]==dig]
      X = rows["x"].to_list()
      Y = rows["y"].to_list()
      labels = rows["label"].to_list()
      plt.scatter(X, Y, label=f"{dig}")
    plt.legend()
    plt.savefig(output_filename)

create_plot("csv/latent_space_ae.csv", "csv/ae.png")
create_plot("csv/latent_space_vae.csv", "csv/vae.png")
