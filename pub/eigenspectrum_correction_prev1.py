"""
Ziele:
+ Daten generieren
+ Nichtmetrisches Distanzmaß benutzen
+ FLIP/CLIP/SHIFT Korrektur auf Similarities ausführen
+ Neue Punkte ebenfalls korrigieren
+ Prüfen, wie gut er die Punkte findet
"""

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
    p = 0.5
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

    print("S_mm", S_mm)
    w, v = np.linalg.eigh(S_mm)
    v = np.flip(v, 1)
    w = w[::-1]
    w[w < 0] = 0  # clip
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


Diss = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
sm, fDis2K = correctMatrix(Diss)
sm[np.abs(sm) < 1e-8] = 0
print("sm", sm)

# Convert new distances
dist = np.array([[4, 5, 6], [9, 8, 7], ])
c = fDis2K(dist)
c[np.abs(c) < 1e-8] = 0
print(c)



# Wenn der Vektor in Similarities umgewandelt wurde, kann ich ihn korrigieren.
# TODO: Wieder in Dissimilarities umwandeln
# %%
# Prüfen, wie gut der Klassifikator die Punkte jetzt findet
# TODO: Über alle Punkte iterieren und den Test machen.


# %%
# Test der Klassifikation OHNE Korrektur
from sklearn.neighbors import NearestNeighbors

classifier = NearestNeighbors(n_neighbors=5, metric=distLP)
classifier.fit(X_train)

predictedNeighbours = classifier.kneighbors(X_test, 1, False)
predictedClasses = Y_train[predictedNeighbours.T]

differences = np.abs(Y_test - predictedClasses).sum()
print("Wrong classified: ", differences)  # 33
print("Correct classified: ", test_samples - differences)  # 217
