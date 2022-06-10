import argparse
import numpy as np
import cv2, os, time
from meta import sample_filepath
from nmf import accelerated_HALS, accelerated_MU
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
from IPython.display import clear_output

nmf = "HALS_Accel_GIFL60"
output_recon_images = False
r = "/mnt/i/dataset-processed"
filepaths = os.listdir(r)
filepaths = [os.path.join(r, f) for f in filepaths if f.split(".")[-1].lower() in ["jpeg","jpg","png"]]

sample_indices = [] 
for idf, f in enumerate(filepaths):
    for i in range(len(sample_filepath)):
        if sample_filepath[i] == f.split("/")[-1]:
            sample_indices.append(idf)

# np.random.seed(42)
# W = np.random.rand(112*112, 60)
# H = np.random.rand(60, 10001)
# X_reconst = np.matmul(W,H)
# L = W.shape[1]
# fig, axes = plt.subplots(L//10,10, figsize=(20, L//5))
# axes = axes.ravel()
# for i in range(len(axes)):
#     axes[i].imshow(W[:,i].reshape((112,112)), cmap='gray')
#     axes[i].axis('off')
# plt.subplots_adjust(wspace=0.1, hspace=0)
# fig.tight_layout()
# fig.savefig(nmf + f"/images/components/NMF_components_{0}.png")

# fig, axes = plt.subplots(5,10, figsize=(20,10))
# axes = axes.ravel()
# for i in range(len(axes)):
#     axes[i].imshow(X_reconst[:,sample_indices[i]].reshape((112,112)), cmap="gray")
#     axes[i].axis('off')
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.tight_layout()
# fig.savefig(nmf + f"/images/reconst/Sample_reconstruction_{0}.png")

# # generate plots
# for inmf in range(1,101):
#     W = np.load(nmf + f"/weights/matrix_W_{inmf}.npy")
#     H = np.load(nmf + f"/weights/matrix_H_{inmf}.npy")
#     X_reconst = np.matmul(W,H)
#     L = W.shape[1]
#     fig, axes = plt.subplots(L//10,10, figsize=(20, L//5))
#     axes = axes.ravel()
#     for i in range(len(axes)):
#         axes[i].imshow(W[:,i].reshape((112,112)), cmap='gray')
#         axes[i].axis('off')
#     plt.subplots_adjust(wspace=0.1, hspace=0)
#     fig.tight_layout()
#     fig.savefig(nmf + f"/images/components/NMF_components_{inmf}.png")

#     fig, axes = plt.subplots(5,10, figsize=(20,10))
#     axes = axes.ravel()
#     for i in range(len(axes)):
#         axes[i].imshow(X_reconst[:,sample_indices[i]].reshape((112,112)), cmap="gray")
#         axes[i].axis('off')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     fig.tight_layout()
#     fig.savefig(nmf + f"/images/reconst/Sample_reconstruction_{inmf}.png")

import imageio
images = []
images.append(imageio.imread(nmf + f"/images/components/NMF_components_{0}.png"))
images.append(imageio.imread(nmf + f"/images/components/NMF_components_{0}.png"))
images.append(imageio.imread(nmf + f"/images/components/NMF_components_{0}.png"))
for i in range(1,30):
    images.append(imageio.imread(nmf + f"/images/components/NMF_components_{i}.png"))
imageio.mimsave(nmf + f"/NMF_components.gif", images, duration=0.33)

images = []
images.append(imageio.imread(nmf + f"/images/reconst/Sample_reconstruction_{0}.png"))
images.append(imageio.imread(nmf + f"/images/reconst/Sample_reconstruction_{0}.png"))
images.append(imageio.imread(nmf + f"/images/reconst/Sample_reconstruction_{0}.png"))
for i in range(1,30):
    images.append(imageio.imread(nmf + f"/images/reconst/Sample_reconstruction_{i}.png"))
imageio.mimsave(nmf + f"/Sample_reconstruction.gif", images, duration=0.33)