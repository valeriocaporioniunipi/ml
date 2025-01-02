'''
Regression
'''

import numpy as np
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler
from utils import get_data, abs_path, target_distribution, mean_euclidean_error_scorer, euclidean_error_scorer
from matplotlib import pyplot as plt



#@ignore_warnings(category=ConvergenceWarning)
def regression(features, targets, n_splits, model):

    """
    Performs linear regression (using sklearn) with k-fold cross-validation with a
    specified number of splits on the given dataset and
    prints evaluation metrics of the linear regression model
    such as MAE (mean absolute error), MSE (mean squared error) and R-squared. 
 
    :param features: features
    :type features: numpy.ndarray
    :param targets: array containing target feature
    :type targets: numpy.narray 
    :param n_splits: number of folds for cross-validation
    :type n_splits: int

    :returns: A tuple containing:
    
        - **best_model** (*sequential*): the best model selected across k-folding.
        - **mae** (*float*): the mean absolute error mean across folds.
        - **mse** (*float*): the mean squared error mean across folds
        - **r2** (*float*): the coefficient of determination mean across folds.
    :rtype: tuple(sequential, float, float, list)

    """

    # Initialize data standardization (done after the k-folding split to avoid leakage)
    scaler = StandardScaler()

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle = True, random_state= 42)

    # Initialize lists to store evaluation metrics and prediction-actual-difference list
    mee_scores = np.zeros(targets.shape[1])
    mee_scores_mean = []

    # Initialization in order to find the best model parameters
    best_model = None
    mee_mean_best = float('inf')

    #Figure initialization:
    _, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Perform k-fold cross-validation
    for _, (train_index, test_index) in enumerate(kf.split(features), 1):
        # Split data into training and testing sets
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Initialize and fit linear regression model
        if model == "l":
            logger.info("Selecting linear regression...")
            model = LinearRegression()
        if model == "g":
            logger.info("Selecting gaussian regression...")
            kernel = C(1.0, (1, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1, 1e2))
            model = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 5)
        if model == "k":
            logger.info("Selecting k-nn regression...")
            model = KNeighborsRegressor()
        model.fit(x_train, y_train)
        # Predict on the test set
        y_pred = model.predict(x_test)

        # Evaluate the model
        mee_mean = mean_euclidean_error_scorer(y_test, y_pred)
        mee = euclidean_error_scorer(y_test, y_pred)

        #Using mse as evaluation metric for best model selection
        if mee_mean < mee_mean_best:
            mee_mean_best = mee_mean
            best_model = model

        mee_scores = np.append(mee_scores, mee)
        mee_scores_mean = np.append(mee_scores_mean, mee_mean)

        #Nested loop for the creation of plots
        for i in range(3): 
            for j in range(3): 
                axes[i, j].plot(y_test[:,j] - y_pred[:,j], y_test[:,i] - y_pred[:,i], marker='o', markersize=2, linestyle='')
                axes[i, j].grid(True)

    # Print average evaluation metrics over all folds
    mee, mee_mean = mee_scores.mean(0), mee_scores_mean.mean(0)

    print("Mean Euclidean Error:", mee)
    #print("Mean Squared Error:", mee_mean) come fa ad essere diverso?

    logger.info("Regression correctly terminated")
    return best_model


#Getting the absolute path to te file through utils function abs_path 
filepath = abs_path("ML-CUP24-TR.csv", "data")

#Extracting features and targets from csv
features, targets = get_data(filepath)

#Getting some targets informations
target_distribution(targets, show = False)

#Performing linear regression:
regression(features, targets, 5, "k")

plt.tight_layout()
plt.show()