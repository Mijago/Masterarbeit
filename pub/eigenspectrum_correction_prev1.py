"""
Ziele:
+ Daten generieren
+ Nichtmetrisches Distanzmaß benutzen
+ FLIP/CLIP/SHIFT Korrektur auf Similarities ausführen
+ Neue Punkte ebenfalls korrigieren
+ Prüfen, wie gut er die Punkte findet
"""


def sim2dis(sim):
    (n, m) = sim.shape

    if n != m:
        raise Exception('The similarity matrix must be square.')

    d1 = np.diag(sim)
    d2 = d1.T
    dis = (d1 + d2) - 2 * sim
    dis = 0.5 * (dis + dis.T)
    return dis


# %%
# 1: Daten generieren

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.spatial.distance

# Specify train and test size
train_samples = 1000
test_samples = 250
# X, Y = datasets.make_blobs(n_samples=train_samples + test_samples, n_features=2, random_state=44)
X, Y = datasets.make_classification(n_samples=train_samples + test_samples, n_features=2, n_redundant=0,
                                    n_informative=2, random_state=44, n_clusters_per_class=2)

X_train, Y_train = X[0:train_samples], Y[0:train_samples]
X_test, Y_test = X[train_samples:], Y[train_samples:]

# Plot the data

plt.title("Generated training dataset", fontsize='small')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.show()


# %%
# 2. Distanzen mit einem nichtmetrischen Abstandsmaß berechnen
def distLP(x1, x2):
    x1 = x1[0:2]
    x2 = x2[0:2]
    diff = np.abs((x1 - x2)[0:2])
    # p = 0.5
    p = 2  # metric euclidean
    n = 0
    for num in diff:
        n += math.pow(num, p)
    n = math.pow(n, 1 / p)
    return n


dissimilarities = scipy.spatial.distance.pdist(X_train, metric=distLP)
dissimilarities = scipy.spatial.distance.squareform(dissimilarities)


# %%
# Flip/clip/shift als Korrektur ausführen
def sum(a, b=0):
    return np.array([np.sum(a, b)])


def correctMatrix(D, correction='clip'):
    # dissimilarities
    N = D.shape[0]

    PInvDmm = np.linalg.pinv(D)
    S_mm = -.5 * (
            D
            - 1 / N * np.ones((N, 1)) * sum(D, 0)
            - 1 / N * sum(D, 0).T @ np.ones((1, N))
            + 1 / pow(N, 2) * np.sum(sum(D @ PInvDmm, 0) @ D.T)
    )
    # Eigval correction
    S_mm = .5 * (S_mm + S_mm.T)

    tmp = sum(D @ PInvDmm, 0)
    center_estimates = tmp / N
    overall_mean_dissimilarity = np.sum(tmp @ D.T) / pow(N, 2)
    mean_dissimilarity = sum(D, 0) / N

    w, v = np.linalg.eigh(S_mm)
    v = np.flip(v, 1)
    w = w[::-1]
    # w[w < 0] = 0  # clip (currently disabled)
    w = np.diag(w)

    S_nm_ = v @ (w @ v.T)
    S_mm_ = S_nm_
    S_mm_ = 0.5 * (S_mm_ + S_mm_.T)

    P = np.linalg.pinv(S_mm_, 1e-4) @ S_nm_

    def fDis2K(DX_km):
        _N, _m = DX_km.shape
        Snm = -.5 * (
                DX_km
                - np.ones((_N, 1)) * mean_dissimilarity
                - (center_estimates @ DX_km.T).T * np.ones((1, _m))
                + overall_mean_dissimilarity
        )
        return Snm @ P

    return S_nm_, fDis2K


# How to convert them back to similarities?

# %%
# Neue Punkte ebenfalls korrigieren
# Nachdem ich die Distanzen von einem Punkt zu allen anderen Punkten berechnet habe, muss ich ihn ebenfalls in SIM
# umwandeln, anschließend korrigieren und wieder in DIS umwandeln.

Diss = dissimilarities
corrected_similarity, fDis2K = correctMatrix(Diss)
corrected_similarity[np.abs(corrected_similarity) < 1e-8] = 0

corrected_dissimilarity = sim2dis(corrected_similarity)

# %%
# Wenn der Vektor in Similarities umgewandelt wurde, kann ich ihn korrigieren.
#  Wieder in Dissimilarities umwandeln


# %%
# Prüfen, wie gut der Klassifikator die Punkte jetzt findet
# TODO: Über alle Punkte iterieren und den Test machen.


# %%
# Test der Klassifikation OHNE Korrektur
from sklearn.neighbors import NearestNeighbors


def dlookup(p1, p2):
    if p1[2] >= 0 and p2[2] >= 0:
        return corrected_dissimilarity[int(p1[2]), int(p2[2])]
    elif p2[2] >= 0:
        otherPtId = int(p2[2])
        dist = np.array([[distLP(p1, n) for n in X_train]])
        local_similarity = fDis2K(dist)

        own_similarity = 1.5
        # TODO The following line is a test, only to test if it can predict the train values again
        # own_similarity = corrected_similarity[otherPtId, otherPtId]

        # revert to dist
        dists = np.array(
            [corrected_similarity[otherPtId, otherPtId] + own_similarity - 2 * local_similarity[0, n] for n in
             range(0, local_similarity.shape[1])])
        return dists[otherPtId]
    else:
        raise Exception()


from vptree2 import VPTree
dtrain = [[n[0], n[1], m] for m, n in enumerate(X_train)]
vp = VPTree(dtrain, dlookup)

# %%
dtest = np.array([[n[0], n[1], -1] for m, n in enumerate(X_train[0:500])])
stest = Y_train[0:500]
correct = 0

for n, v in enumerate(dtest):
    originalclazz = stest[n]
    predictedNeighbour = vp.get_nearest_neighbor(v)[1][2]
    predictedClazz = Y_train[int(predictedNeighbour)]
    if predictedClazz == originalclazz:
        correct += 1
    else:
        print(n)

print("Wrong classified: ", stest.shape[0] - correct)  # 33
print("Correct classified: ", correct)  # 217
