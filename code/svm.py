
import time

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from utils import scorer, get_data, mean_euclidean_error, get_outer, abs_path, w_csv, euclidean_error, standardize_data

def model_selection(features, targets):
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    svr = SVR()
    model = MultiOutputRegressor(svr)

    # define the grid search parameters
    epsilon = np.arange(start=0.1, stop=0.9, step=0.1)
    epsilon = [float(round(i, 4)) for i in list(epsilon)]

    param_grid = [{'estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                   'estimator__gamma': [1e-1, 1e-2, 1e-3, 1e-4, 'auto', 'scale'],
                   'estimator__C': [1e-3, 1e-2, 1e-1, 1, 10, 100],
                   'estimator__degree': [2, 3, 4],
                   'estimator__epsilon': epsilon}]

    start_time = time.time()
    logger.info("Starting theGrid Search...")

    #TODO select the number of active processors w n_jobs; -1 means that all processors are used
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, cv=10,
                        return_train_score=True, scoring=scorer, verbose=1)

    grid_result = grid.fit(features, targets)

    elapsed_time = time.time() - start_time
    logger.info(f"Grid Search ended in {elapsed_time} s")

    means_train = abs(grid_result.cv_results_['mean_train_score'])
    means_test = abs(grid_result.cv_results_['mean_test_score'])
    times_train = grid_result.cv_results_['mean_fit_time']
    times_test = grid_result.cv_results_['mean_score_time']
    params = grid_result.cv_results_['params']

    for m_ts, t_ts, m_tr, t_tr, p in sorted(zip(means_test, times_test, means_train, times_train, params)):
        #print("{} \t TR {:.4f} (in {:.4f}) \t TS {:.4f} (in {:.4f})".format(p, m_tr, t_tr, m_ts, t_ts))
        pass

    logger.info("Best: {:.4f} using {}\n".format(abs(grid.best_score_), grid_result.best_params_))

    return grid_result.best_params_

def predict(model, features_test, targets_test, features_outer):

    targets_pred = model.predict(features_test)

    fig1, ax1 = plt.subplots(1, 3)
    for i in range(3):
        ax1[i].scatter(targets_test[:, i], targets_pred[:, i], marker = 'o')
        ax1[i].set(xlabel="targets test", ylabel="targets predict")
        ax1[i].set_title(f"Variable number {i}")

    # predict on internal test set of data
    internal_test_loss = euclidean_error(targets_test, targets_pred)
    # predict on an outer test set
    targets_outer_pred = model.predict(features_outer)
    # return prediction on outer test set and loss on internal test set
    return targets_outer_pred, internal_test_loss


def plot_learning_curve(model, features, targets, savefig=False):

    # dictify model's parameters
    p = model.get_params()
    params = dict(kernel=p['estimator__kernel'], C=p['estimator__C'],
                  gamma=p['estimator__gamma'], eps=p['estimator__epsilon'])

    # plot learning curve by training and scoring the model for different train sizes
    #TODO select the number of active processors w n_jobs; -1 means that all processors are used
    train_sizes, train_scores_svr, test_scores_svr = \
        learning_curve(model, features, targets, train_sizes=np.linspace(0.1, 1, 60),
                       n_jobs=2, scoring=scorer, cv=10, verbose=1)

    fig2 = plt.figure()
    plt.plot(train_sizes, np.mean(np.abs(train_scores_svr), axis=1))
    plt.plot(train_sizes, np.mean(np.abs(test_scores_svr), axis=1), ls= 'dashed')
    plt.xlabel("train size", fontsize = 20)
    plt.ylabel("loss", fontsize = 20)
    plt.legend(['loss TR', 'loss VL'], fontsize = 20)
    plt.title(f'SVR learning curve \n {params}', fontsize = 20)

    if savefig:
        plt.savefig("plot/sklearnSVM.pdf", transparent = True)

    plt.show()


def sklearn_svm(ms=True):
    logger.info("Initializing SVM...")

    filepath = abs_path("ML-CUP24-TR.csv", "data")

    # read training set
    features, targets, features_test, targets_test = get_data(filepath, split = True)
    features, features_test, targets, targets_test = standardize_data(features, features_test, targets, targets_test)
    # choose model selection or hand-given parameters
    if ms:
        logger.info("Choosing hyperparameters with a GridSearch")
        params = model_selection(features, targets)
    else:
        params = dict(estimator__kernel='rbf', estimator__C=10, estimator__epsilon=0.1, estimator__gamma=0.01)
        logger.info(f"Parameters have been chosen manually: {params}")

    # create model and fit the model
    svr = SVR(kernel=params['estimator__kernel'], C=params['estimator__C'],
              gamma=params['estimator__gamma'], epsilon=params['estimator__epsilon'])

    # we use MOR to perform the multi-output regression task
    model = MultiOutputRegressor(svr)

    # simulating losses on a training and validation set
    features_train, features_val, targets_train, targets_val = train_test_split(features, targets, test_size=0.1)
    model.fit(features_train, targets_train)
    train_losses = mean_euclidean_error(targets_train, model.predict(features_train))
    val_losses = mean_euclidean_error(targets_val, model.predict(features_val))

    # fitting on the entire development set
    model.fit(features, targets)

    # predictions on external test and model assessment
    targets_outer_pred, internal_test_losses = predict(model=model,
                     features_outer=get_outer(abs_path("ML-CUP24-TS.csv", "data")),
                     features_test=features_test, targets_test=targets_test)

    print("TR Loss: ", np.mean(train_losses))
    print("VL Loss: ", np.mean(val_losses))
    print("TS Loss: ", tf.reduce_mean(internal_test_losses).numpy())

    logger.info("Sklearn end.")

    plot_learning_curve(model, features, targets, savefig=True)
    w_csv(targets_outer_pred)


if __name__ == '__main__':
    sklearn_svm(ms = True)