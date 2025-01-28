import time
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve

from utils import monk_data, abs_path, scorer, mean_euclidean_error, euclidean_error, standardize_data

def model_selection(features, targets):
    """
    Perform model selection using Grid Search to find the best hyperparameters for an SVC model.

    :param features: input features for the model
    :type features: numpy.ndarray or pandas.DataFrame
    :param targets: actual target values for evaluation
    :type targets: numpy.ndarray or pandas.DataFrame
    
    :return: best hyperparameters obtained from the grid search
    :rtype: dict
    """
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
    start_time = time.time()

    #TODO select the number of active processors w n_jobs; -1 means that all processors are used
    grid = GridSearchCV(estimator=classifier, param_grid=SVC_params, n_jobs=2, cv=10,
                        return_train_score=True, scoring='accuracy', verbose=1)

    grid_results = grid.fit(features, targets)

    end_time =  time.time() 
    elapsed_time = end_time - start_time
    logger.info(f"Grid Search ended successfully in {elapsed_time}")

    # means_train = abs(grid_results.cv_results_['mean_train_score'])
    # means_test = abs(grid_results.cv_results_['mean_test_score'])
    # times_train = grid_results.cv_results_['mean_fit_time']
    # times_test = grid_results.cv_results_['mean_score_time']
    # params = grid_results.cv_results_['params']

    #for m_ts, t_ts, m_tr, t_tr, p in sorted(zip(means_test, times_test, means_train, times_train, params)):
    #    print("{} \t TR {:.4f} (in {:.4f}) \t TS {:.4f} (in {:.4f})".format(p, m_tr, t_tr, m_ts, t_ts))

    return grid_results.best_params_

def predict(model, features, targets):
    """
    Predict the target values using the given model and compute the evaluation metrics.

    :param model: trained model to make predictions
    :type model: Keras Sequential model or any other suitable trained model
    :param features: input features for prediction
    :type features: numpy.ndarray or pandas.DataFrame
    :param targets: actual target values for evaluation
    :type targets: numpy.ndarray or pandas.DataFrame
    
    :return: predicted values, Mean Squared Error (MSE) and accuracy of the model
    :rtype: tuple (numpy.ndarray, float, float)
    """
    predictions = model.predict(features)
    MSE = mean_squared_error(targets, predictions)
    accuracy = accuracy_score(targets, predictions)
    return predictions, MSE, accuracy

def plot_combined_curves(model, x, y, x_test, y_test, savefig=False):
    """
    Plot the learning curve and accuracy curve side by side for the given model.

    :param model: trained model to evaluate
    :type model: scikit-learn estimator (e.g., SVC)
    :param x: input features for training
    :type x: numpy.ndarray or pandas.DataFrame
    :param y: target values for training
    :type y: numpy.ndarray or pandas.DataFrame
    :param x_test: input features for testing
    :type x_test: numpy.ndarray or pandas.DataFrame
    :param y_test: target values for testing
    :type y_test: numpy.ndarray or pandas.DataFrame
    :param savefig: optional (default = False): whether to save the plot
    :type savefig: bool
    
    :return: None
    """
    p = model.get_params()
    params = dict(kernel=p['kernel'], C=p['C'],
                 gamma=p['gamma'], class_weight=p['class_weight'])
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Learning Curve (Left Plot)
    train_sizes, train_scores_svc, val_scores_svc = learning_curve(
        model, x, y, 
        train_sizes=np.linspace(0.1, 1, 60),
        scoring="neg_mean_squared_error",
        n_jobs=2,
        cv=62,
        verbose=1
    )
    
    ax1.plot(train_sizes, np.mean(np.abs(train_scores_svc), axis=1))
    ax1.plot(train_sizes, np.mean(np.abs(val_scores_svc), axis=1), linestyle="dashed")
    ax1.set_xlabel("Train Size")
    ax1.set_ylabel("Loss")
    ax1.legend(['Loss TR', 'Loss VL'])
    ax1.set_title(f'SVC learning curve for Monk 1 problem with \n {params}')
    
    # Accuracy Curve (Right Plot)
    train_sizes, train_scores_svc, val_scores_svc = learning_curve(
        model, x, y,
        train_sizes=np.linspace(0.1, 1, 60),
        n_jobs=2,
        cv=62,
        verbose=1
    )
    
    test_accuracies = []
    for size in train_sizes:
        model.fit(x[:int(size)], y[:int(size)])
        y_test_pred = model.predict(x_test)
        test_accuracies.append(accuracy_score(y_test, y_test_pred))
    
    ax2.plot(train_sizes, np.mean(np.abs(train_scores_svc), axis=1))
    ax2.plot(train_sizes, np.abs(test_accuracies), linestyle="dashed")
    ax2.set_xlabel("Train Size")
    ax2.set_ylabel("Accuracy")
    ax2.legend(['Accuracy DV', 'Accuracy TS'])
    ax2.set_title(f'SVC accuracy for Monk 1 problem with \n {params}')

    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    if savefig:
        plt.savefig("plot/svc_combined_plots.pdf", transparent=True, bbox_inches='tight')
    
    plt.show()

def modeling_svm():
    """
    Perform modeling using an SVM classifier, evaluate the model, and generate learning and accuracy plots.

    :return: None
    :rtype: None
    """

    # Encoder
    encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()

    # Getting the path to the file
    data_path_train = abs_path('monks-3.train', 'data')
    data_path_test = abs_path('monks-3.test', 'data')

    # Splitting and encoding the training data
    features, targets = monk_data(data_path_train)
    features = encoder.fit_transform(features)

    # Splitting and encoding the test data
    features_test, targets_test = monk_data(data_path_test)
    features_test = encoder.transform(features_test)

    #features = scaler.fit_transform(features)
    #features_test = scaler.transform(features_test)

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
    plot_combined_curves(svc, features, targets, features_test, targets_test, savefig=True)


if __name__ == '__main__':
    modeling_svm()