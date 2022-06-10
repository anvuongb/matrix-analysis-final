import argparse
import numpy as np
import cv2, os, time
from meta import sample_filepath
from nmf import accelerated_HALS, accelerated_MU
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
from IPython.display import clear_output

def get_args():
    parser = argparse.ArgumentParser(description="Accelerated MU/HALS",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datadir", type=str, default=None,
                        help="path to directory containing standardized data")
    parser.add_argument("--algorithm", type=str, default=None,
                        help="NMF algorithm to use, MU or HALS")
    parser.add_argument("--accelerated", action='store_true',
                        help="whether or not to use the accelerated version True/False")
    parser.add_argument("--update-func", type=str, default="gillis",
                        help="which update function to use for hals, `paper` or `gillis`")
    parser.add_argument("--L", type=int, default=30,
                        help="low rank dimension")
    parser.add_argument("--alpha", type=float, default=2,
                        help="control parameter for number of inner loop iterations, ignored if accelerated=False")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="control parameter for stopping condition, ignored if accelerated=False")
    parser.add_argument("--max-iter", type=int, default=2,
                        help="number of iterations to update W and H")
    parser.add_argument("--outputdir", type=str, default=None,
                        help="where to export the output")
    parser.add_argument("--output-recon-images", action='store_true',
                        help="if true, will generate reconstructed images")
    parser.add_argument("--use-sample", action='store_true',
                        help="if true, make use of the list in meta.py to output report")
    parser.add_argument("--restart", action='store_true',
                        help="if true, restart from files available in outputdir")
    parser.add_argument("--save-every", type=int, default=0,
                        help="save weight every # iterations")
    parser.add_argument("--save-with-index", action='store_true',
                        help="save weight every # iterations, without replacement")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    np.random.seed(args.seed)

    print("Loading dataset")
    # Load standardized data
    filepaths = os.listdir(args.datadir)
    filepaths = [os.path.join(args.datadir, f) for f in filepaths if f.split(".")[-1].lower() in ["jpeg","jpg","png"]]
    print(f"Found {len(filepaths)} images from {args.datadir}")

    l = []
    for i in range(len(filepaths)):
        img = np.array(cv2.imread(filepaths[i], cv2.IMREAD_GRAYSCALE))
        img = img.reshape((112*112,1))
        l.append(img)
    X = np.concatenate(l, axis=1)
    del l
    print(f"Dataset loaded, shape={X.shape}")

    if args.algorithm not in ["hals", "mu"]:
        raise Exception("Invalid value of algorithm={args.algorithm}, must be either {hals} or {mu}")

    print(f"Starting to perform NMF with the following params: algorithm={args.algorithm}, max_iter={args.max_iter}, accelerated={args.accelerated}, L={args.L}, alpha={args.alpha}, epsillon={args.eps}, update_func={args.update_func}")
    alpha = 0
    if args.accelerated:
        alpha = args.alpha

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    if not os.path.exists(os.path.join(args.outputdir, "weights")):
        os.makedirs(os.path.join(args.outputdir, "weights"))
    
    if args.output_recon_images:
        if not os.path.exists(os.path.join(args.outputdir, "reconstruc_images")):
            os.makedirs(os.path.join(args.outputdir, "reconstruc_images"))

    def save_cb(W, H, norm_error, e_time, save_path, idx):
        if idx > 0:
            np.save(save_path + "/weights" + f"/matrix_W_{idx}.npy", W)
            np.save(save_path + "/weights" + f"/matrix_H_{idx}.npy", H)
            np.save(save_path + "/weights" + "/norm_error.npy", norm_error)
            np.save(save_path + "/weights" + "/e_time.npy", e_time)
        else:
            np.save(save_path + "/weights" + "/matrix_W.npy", W)
            np.save(save_path + "/weights" + "/matrix_H.npy", H)
            np.save(save_path + "/weights" + "/norm_error.npy", norm_error)
            np.save(save_path + "/weights" + "/e_time.npy", e_time)

    if args.algorithm == "mu":
        W, H, norm_error, e_time = accelerated_MU(X, L=args.L, alpha=alpha, epsillon=args.eps, max_iter=args.max_iter, save_every_iter=args.save_every, save_with_index=args.save_with_index,save_cb=save_cb, weight_path=args.outputdir, restart=args.restart)
    else:
        W, H, norm_error, e_time = accelerated_HALS(X, L=args.L, alpha=alpha, epsillon=args.eps, max_iter=args.max_iter, update_func=args.update_func, save_with_index=args.save_with_index,save_every_iter=args.save_every, save_cb=save_cb, weight_path=args.outputdir, restart=args.restart)
    
    print("NMF done, now performing reconstruction")
    X_reconst = np.matmul(W, H)

    sample_indices = [] 
    if args.use_sample:
        for idf, f in enumerate(filepaths):
            for i in range(len(sample_filepath)):
                if sample_filepath[i] == f.split("/")[-1]:
                    sample_indices.append(idf)
    else:
        sample_indices = list(range(50)) # take first 50 images as sample output

    # save npy weights
    s = args.outputdir + "/weights"
    print(f"Reconstruction done, serializing weights to {s}")
    np.save(args.outputdir + "/weights" + "/matrix_W.npy", W)
    np.save(args.outputdir + "/weights" + "/matrix_H.npy", H)
    np.save(args.outputdir + "/weights" + "/norm_error.npy", norm_error)
    np.save(args.outputdir + "/weights" + "/e_time.npy", e_time)
    # np.save(args.outputdir + "/weights" + "/matrix_X_reconst.npy", X_reconst)

    print("Serialization done, generating plots")
    # generate plots
    fig, axes = plt.subplots(args.L//10,10, figsize=(20, args.L//5))
    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].imshow(W[:,i].reshape((112,112)), cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0)
    fig.tight_layout()
    fig.savefig(args.outputdir + "/NMF_components.png")

    fig, axes = plt.subplots(5,10, figsize=(20,10))
    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].imshow(X_reconst[:,sample_indices[i]].reshape((112,112)), cmap="gray")
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig(args.outputdir + "/Sample_reconstruction.png")

    if args.output_recon_images:
        s = args.outputdir + "/reconstruc_images"
        print(f"Generate plots done, outputing reconstructed images to {s}")
        for i in range(X_reconst.shape[1]):
            img = X_reconst[:,i].reshape((112,112))
            img_path = os.path.join(s, filepaths[i].split("/")[-1])
            print(f"{i} {img_path}")
            cv2.imwrite(img_path, img)
            if i%20==0:
                clear_output(wait=True)
    
    print(f"Everything done, check results at {args.outputdir}")

if __name__ == "__main__":
    main()