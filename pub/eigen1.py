import time

from nystroem.sim2dis import sim2dis
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.spatial.distance
from pub.correct_matrix import correctMatrix
from pub.vptree2 import VPTree

train_samples = 1000
test_samples = 250


def buildDistLp(p=0.5):
    def distLP(x1, x2):
        x1 = x1[0:2]
        x2 = x2[0:2]
        diff = np.abs((x1 - x2)[0:2])
        # p = 0.5
        n = 0
        for num in diff:
            n += math.pow(num, p)
        n = math.pow(n, 1 / p)
        return n

    return distLP


# distance_measure = buildDistLp(2)  # euclidean, metric
distance_measure = buildDistLp(0.001)  # non-metric

corrected_dissimilarity = None
X_train = None
Y_train = None


# Train the VP-Tree
def dlookup(p1, p2):
    if p1[2] >= 0 and p2[2] >= 0:
        return corrected_dissimilarity[int(p1[2]), int(p2[2])]
    elif p2[2] >= 0:
        now = time.time()
        otherPtId = int(p2[2])
        dist = np.array([[distance_measure(p1, n) for n in X_train]])
        corrected_distance = fDis2D(dist, otherPtId)
        return corrected_distance
    else:
        raise Exception()


# X, Y = datasets.make_blobs(n_samples=train_samples + test_samples, n_features=2, random_state=44)
X, Y = datasets.make_classification(n_samples=train_samples + test_samples, n_features=2, n_redundant=0,
                                    n_informative=2, random_state=44, n_clusters_per_class=2)
#X, Y = datasets.make_circles(n_samples=train_samples + test_samples, noise=0.2, factor=0.5, random_state=1)
X_train, Y_train = X[0:train_samples], Y[0:train_samples]
X_test, Y_test = X[train_samples:], Y[train_samples:]

print("Build dissimilarities")
now = time.time()
# Calculate all distances
dissimilarities = scipy.spatial.distance.pdist(X_train, metric=distance_measure)
dissimilarities = scipy.spatial.distance.squareform(dissimilarities)
print("Build dissimilarities - done in ", time.time() - now)

# correct the dissimilarities
#corrected_similarity, fDis2K, fDis2D = correctMatrix(dissimilarities, None)  # circles 50% # class 87%
corrected_similarity, fDis2K, fDis2D = correctMatrix(dissimilarities, "clip")  # circles 48% # class 86%
#corrected_similarity, fDis2K, fDis2D = correctMatrix(dissimilarities, "flip")  # circles 18%
corrected_similarity[np.abs(corrected_similarity) < 1e-8] = 0
print("Corrected Similarities - done in ", time.time() - now)

# Mirror it, so it is s
now = time.time()
i_lower = np.tril_indices(corrected_similarity.shape[0], -1)
corrected_similarity[i_lower] = corrected_similarity.T[i_lower]
corrected_dissimilarity = sim2dis(corrected_similarity)
print("Corrected dissimilarities - done in ", time.time() - now)

dtrain = [[n[0], n[1], m] for m, n in enumerate(X_train)]
vp = VPTree(dtrain, dlookup)
print("Finished Training")

# Prepare data to fill the color array
h = .1  # step size in the mesh
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
all_points_x = np.arange(x_min, x_max, h)
all_points_y = np.arange(y_min, y_max, h)
print("Points: ", all_points_y.shape[0] * all_points_x.shape[0], all_points_x.shape, all_points_y.shape)


# Test the existing points
def predict(point):
    x, y = point
    nearest = vp.get_nearest_neighbor(np.array([x, y, -1]))
    return Y_train[int(nearest[1][2])]


# Load process pool
# This must occour AFTER setting all the used variables!!
from pathos.pools import ProcessPool, ThreadPool

with ProcessPool() as pool:
    now = time.time()
    correct_classified = np.sum(np.array(pool.amap(predict, X_test).get()) == Y_test)
    print("predicted %d of %d correct. -> %d Prozent" %
          (correct_classified, test_samples, correct_classified / test_samples * 100.0))
    print("Tested Train Data in s ", time.time() - now)


# Now predict the other points
def cup(x):
    result = np.zeros(shape=(all_points_y.shape))
    for ky, y in enumerate(all_points_y):
        result[ky] = predict((x, y))
    return result


colors = None
with ProcessPool() as pool:
    now = time.time()
    colors = np.array(pool.amap(cup, all_points_x).get()).T
    print("Generated colors in s ", time.time() - now)

# for kx, x in enumerate(all_points_x):
#   now = time.time()
#   for ky, y in enumerate(all_points_y):
#       nearest = vp.get_nearest_neighbor(np.array([x, y, -1]))
#       colors[ky, kx] = Y_train[int(nearest[1][2])]
#   print("Done with line %d of %d in %ds" % (kx + 1, all_points_x.shape[0], time.time() - now))

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.contourf(all_points_x, all_points_y, colors)
ax.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=Y_train, s=25, edgecolor='k')
plt.title("Classification areas (Vantage Point Tree)")
plt.show()
