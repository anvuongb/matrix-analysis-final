import argparse
import numpy as np
import cv2, os, time
from meta import sample_filepath
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
from IPython.display import clear_output

path = "/mnt/i/dataset-processed"
filepaths = os.listdir(path)
filepaths = [os.path.join(path, f) for f in filepaths if f.split(".")[-1].lower() in ["jpeg","jpg","png"]]
print(f"Found {len(filepaths)} images from {path}")

sample_indices = [] 
for idf, f in enumerate(filepaths):
    for i in range(len(sample_filepath)):
        if sample_filepath[i] == f.split("/")[-1]:
            sample_indices.append(idf)

W = np.load("./MU_L30/weights/matrix_W.npy")
H = np.load("./MU_L30/weights/matrix_H.npy")

X_reconst = np.matmul(W, H)

# generate plots
fig, axes = plt.subplots(3,10, figsize=(20, 6))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].imshow(W[:,i].reshape((112,112)), cmap='gray')
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0)
fig.tight_layout()
fig.savefig("./NMF_components.png")

fig, axes = plt.subplots(5,10, figsize=(20,10))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].imshow(X_reconst[:,sample_indices[i]].reshape((112,112)), cmap="gray")
    axes[i].axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
fig.savefig("./Sample_reconstruction.png")
