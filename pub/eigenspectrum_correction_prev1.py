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

# Definition der dis2sim und sim2dis Funktionen
def isSymmetric(d):
    return not np.abs(d - d.T).any()


def dis2sim(dis):
    (n, m) = dis.shape
    if n != m:
        raise Exception('The dissimilarity matrix must be square.')
    if np.diag(dis).any():
        raise Exception('The dissimilarity matrix must have zero diagonal.')
    if not isSymmetric(dis):
        raise Exception('The dissimilarity matrix must be symmetric.')

    J = np.eye(n) - np.tile(1 / n, (n, n))
    sim = -0.5 * J @ dis @ J
    sim = 0.5 * (sim + sim.T)
    return sim


def sim2dis(sim):
    (n, m) = sim.shape
    if n != m:
        raise Exception('The similarity matrix must be square.')
    if not isSymmetric(sim):
        raise Exception('The similarity matrix must be symmetric.')
    d1 = np.diag(sim)
    d2 = d1.T
    dis = (d1 + d2) - 2 * sim
    dis = 0.5 * (dis + dis.T)
    return dis


# Definition der Korrekturverfahren
def clip_matrix_with_exact_eigenvalue(eigenvalues, eigenvectors):
    eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)
    eigenvalues = np.eye(len(eigenvalues)) * eigenvalues
    matrix = np.linalg.multi_dot([eigenvectors, eigenvalues, eigenvectors.T])
    return matrix


def clip_matrix_with_exact_eigenvalue1(mat, eigenvalues, eigenvectors):
    Vclip = eigenvectors * np.power(np.abs(eigenvalues), -0.5) @ np.diag(eigenvalues > 0)
    return mat @ Vclip @ Vclip.T @ mat, Vclip


similarities = dis2sim(dissimilarities)
sw, sv = np.linalg.eig(similarities)

# Unterscheiden sich nur in der 15. Nachkommastelle
simClipped_orig = clip_matrix_with_exact_eigenvalue(sw, sv)
simClipped, Vclip = clip_matrix_with_exact_eigenvalue1(similarities, sw, sv)

# Spiegeln, da wir minimale Unterschiede in den späteren Nachkommastellen haben,
# so dass die Matrix nicht mehr symmetrisch ist.
i_lower = np.tril_indices(simClipped.shape[0], -1)
simClipped[i_lower] = simClipped.T[i_lower]

disClipped = sim2dis(simClipped)

# %%
# Neue Punkte ebenfalls korrigieren
# Nachdem ich die Distanzen von einem Punkt zu allen anderen Punkten berechnet habe, muss ich ihn ebenfalls in SIM
# umwandeln, anschließend korrigieren und wieder in DIS umwandeln.
# TODO Ich wandle die Distanzen oben mit Double Centering in Similarities um. Wie mache ich das hier mit nur einem Vektor?
# TODO Wie kann ich das mit EINEM Punkt gemacht habe, wie kann ich das mit N Punkten machen?

# Wenn der Vektor in Similarities umgewandelt wurde, kann ich ihn korrigieren.
# st = st * Vclip @ Vclip @ st
# TODO: Wieder in Similarities umwandeln

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
