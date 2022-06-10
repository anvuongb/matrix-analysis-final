import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import seaborn as sns

NMF_algo_list = [
    "HALS_Accel_L30",
    "HALS_Accel_L60",
    "HALS_Accel_L120",
    "HALS_Accel_L180",
    "HALS_Accel_L240",
    "HALS_Accel_L360",
    "HALS_Accel_L480",
    "HALS_Accel_L720"
]

labels = ["L=30","L=60","L=120","L=180","L=240","L=360","L=480","L=720"]

embs = {}
embs["raw"] = np.load("./embeddings/raw.npy")
for nmf in NMF_algo_list:
    embs[nmf] = np.load(f"./embeddings/{nmf}.npy")

def dist(a,b):
    return np.sum(np.square(a-b))

def get_confidence_score(sim_score):
    min_sim=0.4
    max_sim=2.83
    min_confidence=0.01
    sim_score = min(max(sim_score, min_sim), max_sim)
    confidence_score = (-sim_score + max_sim) * (1 - min_confidence) / (max_sim - min_sim) + min_confidence
    return confidence_score

fig = plt.figure(figsize=(8,5))
sns.set(style="darkgrid")
for idx, nmf in enumerate(NMF_algo_list):
    distances = np.sum(np.square(embs["raw"] - embs[nmf]), axis=1).reshape((-1,1))
    scores = np.apply_along_axis(get_confidence_score, 1, distances)
    
    sns.kdeplot(scores.ravel(), shade=True)

plt.legend(labels=labels)
plt.xlabel("score")
plt.tight_layout()
# plt.show()
fig.savefig("embedding_score_density.png")

np.random.seed(42)
random_1000 = np.random.randint(0,10001,1000)
random_corresp = []

for idx, v in enumerate(random_1000):
    r = np.random.randint(0,10001,10)
    while v in r:
        r = np.random.randint(0,10001,10)
    random_corresp.append(r)

fig = plt.figure(figsize=(8,5))
sns.set(style="darkgrid")
for idx, nmf in enumerate(NMF_algo_list + ["raw"]):
    l = []
    for idx, v in enumerate(random_1000):
        x = np.sum(np.square(embs[nmf][v] - embs[nmf][random_corresp[idx]]), axis=1).reshape((-1,1))
        l.append(x)
    distances = np.vstack(l)
    scores = np.apply_along_axis(get_confidence_score, 1, distances)
    
    sns.kdeplot(scores.ravel(), shade=True)

plt.legend(labels=labels + ["raw"])
plt.xlabel("score")
plt.tight_layout()
# plt.show()
fig.savefig("embedding_random_pair_score_density.png")


# these are size of W and H (MB)
storage = {
    "HALS_Accel_L30":2.3+2.9,
    "HALS_Accel_L60":4.6+5.8,
    "HALS_Accel_L120":9.3+11.7,
    "HALS_Accel_L180":14+17.6,
    "HALS_Accel_L240":18.7+23.5,
    "HALS_Accel_L360":28.1+35.2,
    "HALS_Accel_L480":37.5+47,
    "HALS_Accel_L720":56.2+70.5,
}

accuracy_all = {}
accuracy = {}
for idx, nmf in enumerate(NMF_algo_list):
    distances = np.sum(np.square(embs["raw"] - embs[nmf]), axis=1).reshape((-1,1))
    scores = np.apply_along_axis(get_confidence_score, 1, distances)
    accuracy[nmf] = np.average(scores)
    accuracy_all[nmf] = scores

x = np.array(list(storage.values()))/980*100
y = np.array(list(accuracy.values()))

fig = plt.figure(figsize=(8,5))
sns.set(style="darkgrid")
kwargs = dict(linestyle='--', color='r', marker ='o', linewidth=1.2, markersize=13)
sns.lineplot(x=x, y=y,**kwargs)
for i, l in enumerate(labels):
    plt.annotate(l, (x[i]+0.2, y[i]-0.01))
plt.ylabel("mean score")
plt.xlabel("compression ratio (%)")
plt.tight_layout()
# plt.show()
fig.savefig("embedding_compress_ratio_versus_score.png")