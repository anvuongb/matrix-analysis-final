from models.ArcFaceEmbeddingExtractorTRT import ArcFace
import cv2
import numpy as np
import os
import time

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# init model
model = ArcFace(
    trt_serving_host="localhost",
    trt_serving_port=8501
)
# start = time.time()
# print("Loading raw dataset")
# # Load standardized data
# r = "/mnt/i/dataset-processed"
# filepaths = os.listdir(r)
# filepaths = [os.path.join(r, f) for f in filepaths if f.split(".")[-1].lower() in ["jpeg","jpg","png"]]
# print(f"Found {len(filepaths)} images from {r}")

# l = []
# for i in range(len(filepaths)):
#     img = np.array(cv2.imread(filepaths[i]))
#     l.append(img)
# t = time.time()-start
# print(f"Raw dataset loaded, took {t}")

# start = time.time()
# print("Generating raw embeddings")
# list_embs = []
# for i in np.arange(0, len(l), 8):
#     # batch_size = 8
#     embeddings = model.predict(l[i:i+8])
#     list_embs.append(embeddings)
#     if i % 80 == 0:
#         print(f"{i+1} embeddings generated")
# raw_embds = np.vstack(list_embs)
# np.save("./embeddings/raw.npy", raw_embds)
# del raw_embds
# t = time.time()-start
# print(f"Raw embeddings generated, took {t}")

NMF_algo_list = [
    # "HALS_Accel_L30",
    # "HALS_Accel_L60",
    # "HALS_Accel_L120",
    # "HALS_Accel_L180",
    # "HALS_Accel_L240",
    # "HALS_Accel_L360",
    "HALS_Accel_L480",
    "HALS_Accel_L720",
    # "HALS_L30",
    # "HALS_L60",
    # "MU_Accel_L30",
    # "MU_Accel_L60",
    # "MU_L30",
    # "MU_L60"
]

for nmf in NMF_algo_list:
    start = time.time()
    print(f"Generating {nmf} embeddings")
    W = np.load(f"./{nmf}/weights/matrix_W.npy")
    H = np.load(f"./{nmf}/weights/matrix_H.npy")
    X = np.matmul(W, H)
    X = np.repeat(X, 3, axis=0).astype('uint8') # convert to 3 channels by repeating

    l = []
    for i in range(X.shape[1]):
        l.append(X[:,i].reshape((112,112,3)))
    list_embs = []
    for i in np.arange(0,len(l), 8):
        # batch_size = 8
        embeddings = model.predict(l[i:i+8])
        list_embs.append(embeddings)
        if i % 80 == 0:
            print(f"{i+1} embeddings generated")
    embs = np.vstack(list_embs)
    t = time.time()-start
    print(f"{nmf} embeddings generated, took {t}")
    np.save(f"./embeddings/{nmf}.npy", embs)
