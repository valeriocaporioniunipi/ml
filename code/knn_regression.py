'''
KNN_ Regression
'''

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils import get_data, abs_path, target_distribution
from matplotlib import pyplot as plt




def knn_regressor(features, targets, n_splits):

    """
    Performs KNN regression (using sklearn) with k-fold cross-validation with a
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
    mae_scores, mse_scores, r2_scores = [],[],[]

    # Initialization in order to find the best model parameters
    best_model = None
    mse_best = float('inf')

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
        model = KNeighborsRegressor()
        model.fit(x_train, y_train)
        # Predict on the test set
        y_pred = model.predict(x_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test, y_pred)
        #Using mse as evaluation metric for best model selection
        if mse < mse_best:
            mse_best = mse
            best_model = model
        
        #Nested loop for the creation of plots
        for i in range(3): 
            for j in range(3): 
                axes[i, j].plot(y_test[:,j] - y_pred[:,j], y_test[:,i] - y_pred[:,i], marker='o', markersize=2, linestyle='')
                axes[i, j].grid(True)

        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)

    # Print average evaluation metrics over all folds
    mae, mse, r2 = np.mean(mae_scores),np.mean(mse_scores), np.mean(r2_scores)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return best_model, mae, mse, r2


#Getting the absolute path to te file through utils function abs_path 
filepath = abs_path("ML-CUP24-TR.csv", "data")

#Extracting features and targets from csv
features, targets = get_data(filepath)

#Performing gaussian regression:
knn_regressor(features, targets, 5)

#Getting some targets informations
target_distribution(targets, show = True)

plt.tight_layout()
plt.show()