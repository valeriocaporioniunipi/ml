import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve

from utils import monk_data, abs_path, scorer, mean_euclidean_error, euclidean_error

def model_selection(features, targets):
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    classifier = SVC()

    # define the grid search parameters
    SVC_params =  {'kernel' : ['rbf', 'poly', 'sigmoid'], 
              'gamma' : [1, 0.1, 0.01, 0.001],
              'C': [0.1, 1, 10, 100], 
               'class_weight' : [None, 'balanced'],
               'degree' : [2,3]}
    # Folds
    folds = StratifiedKFold(n_splits = 10, shuffle = True, random_state=42)
    
    # Grid Search
    SVC_grid = GridSearchCV(estimator=classifier, 
                        param_grid=SVC_params, 
                        n_jobs=2, 
                        cv=folds,
                        scoring = 'accuracy',
                        verbose=1,
                        return_train_score=True)
    
    # Grid results
    grid_results = SVC_grid.fit(features, targets)
    logger.info("Starting theGrid Search...")

    #TODO select the number of active processors w n_jobs; -1 means that all processors are used
    grid = GridSearchCV(estimator=classifier, param_grid=SVC_params, n_jobs=2, cv=10,
                        return_train_score=True, scoring='accuracy', verbose=1)

    grid_results = grid.fit(features, targets)

    logger.info(f"Grid Search ended successfully")

    # means_train = abs(grid_results.cv_results_['mean_train_score'])
    # means_test = abs(grid_results.cv_results_['mean_test_score'])
    # times_train = grid_results.cv_results_['mean_fit_time']
    # times_test = grid_results.cv_results_['mean_score_time']
    # params = grid_results.cv_results_['params']

    #for m_ts, t_ts, m_tr, t_tr, p in sorted(zip(means_test, times_test, means_train, times_train, params)):
    #    print("{} \t TR {:.4f} (in {:.4f}) \t TS {:.4f} (in {:.4f})".format(p, m_tr, t_tr, m_ts, t_ts))

    return grid_results.best_params_

def predict(model, features, targets):
    predictions = model.predict(features)
    MSE = mean_squared_error(targets, predictions)
    accuracy = accuracy_score(targets, predictions)
    return predictions, MSE, accuracy

def plot_learning_curve(model, x, y, savefig = False):
    p = model.get_params()
    params = dict(kernel=p['kernel'], C=p['C'],
                  gamma=p['gamma'], class_weight = p['class_weight'])
    
    train_sizes, train_scores_svc, val_scores_svc = learning_curve(model, x, y, train_sizes=np.linspace(0.1, 1, 60), scoring="neg_mean_squared_error", n_jobs = 2, cv = 62, verbose=1)

    plt.plot(train_sizes, np.mean(np.abs(train_scores_svc), axis=1))
    plt.plot(train_sizes, np.mean(np.abs(val_scores_svc), axis=1))
    plt.xlabel("train size")
    plt.ylabel("loss")
    plt.legend(['loss TR', 'loss VL'])
    plt.title(f'SVC learning curve \n {params}')

    if savefig:
        plt.savefig("plot/svc_3_learning.pdf", transparent = True)
    plt.show()

def plot_accuracy(model, x, y, x_test, y_test, savefig = False):
    p = model.get_params()
    params = dict(kernel=p['kernel'], C=p['C'],
                  gamma=p['gamma'], class_weight = p['class_weight'])
    
    train_sizes, train_scores_svc, val_scores_svc = learning_curve(model, x, y, train_sizes=np.linspace(0.1, 1, 60), n_jobs = 2, cv = 62, verbose=1)
    test_accuracies = []
    for size in train_sizes:
        model.fit(x[:size], y[:size])
        y_test_pred = model.predict(x_test)
        test_accuracies.append(accuracy_score(y_test, y_test_pred))
    plt.plot(train_sizes, np.mean(np.abs(train_scores_svc), axis=1))
    #plt.plot(train_sizes, np.mean(np.abs(val_scores_svc), axis=1))
    plt.plot(train_sizes, np.abs(test_accuracies))
    plt.xlabel("train size")
    plt.ylabel("accuracy")
    plt.legend(['accuracy DV', 'accuracy TS'])
    plt.title(f'SVC accuracy curve \n {params}')

    if savefig:
        plt.savefig("plot/svc_3_accuracy.pdf", transparent = True)
    plt.show()

def modeling_svm():

    # Encoder
    encoder = OneHotEncoder()

    # Getting the path to the file
    data_path_train = abs_path('monks-3.train', 'data')
    data_path_test = abs_path('monks-3.test', 'data')

    # Splitting and encoding the training data
    features, targets = monk_data(data_path_train)
    features = encoder.fit_transform(features)

    # Splitting and encoding the test data
    features_test, targets_test = monk_data(data_path_test)
    features_test = encoder.transform(features_test)

    # Finding the best parameters through the grid search
    best_parameters = model_selection(features, targets)
    print("Best parameters: ", best_parameters)

    # Building a model with the best parameters
    svc = SVC(**best_parameters, random_state = 42)

    # simulating losses on a training and validation set
    features_train, features_val, targets_train, targets_val = train_test_split(features, targets, test_size=0.1)
    svc.fit(features_train, targets_train)
    train_losses = mean_squared_error(targets_train, svc.predict(features_train))
    val_losses = mean_squared_error(targets_val, svc.predict(features_val))

    # fitting on the entire development set
    svc.fit(features, targets)
    development_accuracy= accuracy_score(targets, svc.predict(features))
    
    # predicting on test set
    pred_test, test_loss, test_accuracy = predict(model = svc, features = features_test, targets = targets_test)

    print("Training loss: ", np.mean(train_losses))
    print("Validation loss: ", np.mean(val_losses))
    print("Test loss: ", test_loss)

    print("Development accuracy: ", development_accuracy)
    print("Test accuracy: ", test_accuracy)

    print(classification_report(pred_test, targets_test))

    svc = SVC(**best_parameters, random_state = 42)
    plot_learning_curve(svc, features, targets, savefig=True)
    plot_accuracy(svc, features, targets, features_test, targets_test, savefig=True )


if __name__ == '__main__':
    modeling_svm()