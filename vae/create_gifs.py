import imageio.v2 as imageio
import os

def creategif(dirpath, dim):
    images = []
    files = os.listdir(f"{dirpath}/{dim}")
    files.sort(key=lambda x: int(x.split('.')[0]))
    images = [imageio.imread(f"{dirpath}/{dim}/{file}") for file in files]
    imageio.mimsave(f"{dirpath}/{dim}.gif", images)

def creategifs(dirpath):
    dirs = next(os.walk(dirpath))[1]
    for dim in dirs:
        creategif(dirpath, dim)

creategifs("traverse/vae/imgs")
creategifs("traverse/ae/imgs")
