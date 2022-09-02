import os
import numpy as np
import sys

# print(os.getcwd())
# sys.path.append("..")

from FaST_LMM import FASTLMM

n, p = 1000, 5
np.random.seed(5)
X = np.random.normal(0, 1, size=[n, p])
n_clusters = 10
W = np.zeros([n, n_clusters])
for i in range(n_clusters):
    cluster_size = n // n_clusters

    W[(i * cluster_size):((i + 1) * cluster_size - 1), i] = 1

beta = np.random.normal(0, 20, p)
sigma_g2 = 3
delta = 2
sigma_e2 = delta * sigma_g2
print('sigma_g2: ', sigma_g2)
print('sigma_e2: ', sigma_e2)
print('beta: ', beta)

epsilon = np.random.normal(0, np.sqrt(sigma_e2), n)
u = np.random.normal(0, np.sqrt(sigma_g2), n_clusters)
z = W @ u

y = X @ beta + z + epsilon

data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
np.savetxt("./data/testData.csv", data, delimiter=",")
#####################################

f = FASTLMM(False, REML=True)

f.fit(X, y, W)
print(f.delta_temp)
f.plot_likelihood(REML=True)

print(f.beta)
