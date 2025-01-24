import numpy as np
from matplotlib import pyplot as plt
from utils import torch_mee, abs_path, get_data, get_outer, w_csv, mean_euclidean_error

filepath = abs_path("ML-CUP24-TR.csv", "data")
features, targets, features_test, targets_test = get_data(filepath, split = True)

n_bins = 100

fig, ax = plt.subplots(4,3, tight_layout = True )
for i in range(3):
    ax[0][i].hist(features[i], bins = n_bins, color = 'k')
    ax[1][i].hist(targets[i], bins = n_bins, color = 'r')
    ax[2][i].hist(features_test[i], bins = n_bins, color = 'b')
    ax[3][i].hist(targets_test[i], bins = n_bins, color = 'y')
plt.show()

print(mean_euclidean_error(np.array([1.,2., 3.]), np.array([1., 3., 3.])))