import time
import numpy as np
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plt
from keras import Sequential
from keras import optimizers
from keras import metrics
from keras import layers
from keras import regularizers

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from scikeras.wrappers import KerasClassifier

from utils import monk_data, abs_path
 
seed = 42
np.random.seed(seed)

def create_nn(input_shape,
                  hidden_nodes = 16,
                  summary = False,
                  activation = 'relu',
                  eta = 0.002,
                  alpha=0.7,
                  lmb = 0.0001,
                  init_mode = 'glorot_normal'
                  ):
    """
    Create a simple feedforward neural network using Keras.

    This function creates a neural network with a single hidden layer and compiles it 
    using a stochastic gradient descent optimizer. The network is designed for binary 
    classification tasks.

    :param input_shape: The shape of the input data.
    :type input_shape: int
    :param hidden_nodes: Number of neurons in the hidden layer (default: 16).
    :type hidden_nodes: int
    :param summary: Whether to display the model summary (default: False).
    :type summary: bool
    :param activation: Activation function to use in the hidden layer (default: 'relu').
    :type activation: str
    :param eta: Learning rate for the stochastic gradient descent optimizer (default: 0.002).
    :type eta: float
    :param alpha: Momentum for the stochastic gradient descent optimizer (default: 0.7).
    :type alpha: float
    :param lmb: L2 regularization parameter for the kernel weights (default: 0.0001).
    :type lmb: float
    :param init_mode: Initialization method for the kernel weights (default: 'glorot_normal').
    :type init_mode: str

    :return: A compiled Keras Sequential model.
    :rtype: keras.models.Sequential
    """
    model = Sequential()

    # Input layer 
    model.add(layers.Dense(units=hidden_nodes, activation=activation,
                            input_shape=(input_shape,),
                            kernel_initializer=init_mode,
                            kernel_regularizer=regularizers.L2(lmb)))

    # Output layer
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # Adding optimizer for the model
    optimizer = optimizers.SGD(learning_rate=eta, momentum=alpha)

    # Compiling the model
    model.compile(optimizer=optimizer, loss = 'mean_squared_error', metrics=['accuracy'])

    # Printing summary, if specified
    if summary:
        logger.info("Model successfully compiled, showing detailed summary ")
        model.summary()
    else: 
        pass
    return model

def model_selection(x, y, n_splits, epochs):

    """
    Perform hyperparameter optimization using grid search with cross-validation.

    This function utilizes Keras models wrapped with scikeras for compatibility with 
    scikit-learn's GridSearchCV, allowing efficient exploration of hyperparameter combinations 
    to optimize model performance.

    :param features: Input features for training.
    :type features: numpy.ndarray
    :param targets: Target values for training.
    :type targets: numpy.ndarray
    :param n_splits: Number of folds for K-Fold cross-validation.
    :type n_splits: int
    :param epochs: Number of epochs for training the model.
    :type epochs: int

    :return: Best hyperparameters identified by grid search.
    :rtype: dict
    """
    
    
    input_shape = np.shape(x)[1]
    
    model = KerasClassifier(model=create_nn, input_shape = input_shape, epochs=epochs, batch_size= 25, verbose=0)

    # Setting the grid search parameters
    eta = [0.5, 0.05, 0.005]

    alpha = [0.1, 0.4, 0.7, 1.0]

    lmb = [0, 0.1, 0.01]

    param_grid = dict(model__eta=eta, model__alpha=alpha, model__lmb=lmb)

    # K-folding definition
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', refit = 'accuracy',
        cv = folds, n_jobs = 8, return_train_score=True, verbose = 0)

    # Fitting grid search
    start_time = time.time()
    logger.info("Starting Grid Search for hyperparameter optimization")
    grid_result = grid.fit(x, y)

    end_time =  time.time() 
    elapsed_time = end_time - start_time
    logger.info(f"Grid search concluded in {elapsed_time}")
    best_params = grid_result.best_params_

    # Summarizing results
    logger.info(f"Best: {grid_result.best_score_} using {best_params}")
    
    return best_params

def predict(model, features_test, targets_test):
    """
    Make predictions using the trained model and evaluate performance on test data.

    :param model: trained Keras model used for predictions
    :type model: Sequential
    :param features_test: features of the internal test set
    :type features_test: numpy.ndarray
    :param targets_test: true target values of the internal test set
    :type targets_test: numpy.ndarray
    :param features_outer: features of the outer test set for final predictions
    :type features_outer: numpy.ndarray

    :return: predictions on the outer test set and the loss on the internal test set
    :rtype: tuple
    """
    # Prediction
    targets_pred = model.predict(features_test)

    # Reshape targets_pred per match con targets_test
    targets_pred = targets_pred.flatten()  
    
    # Computing the mean_squared_error
    test_loss = mean_squared_error(targets_test, targets_pred)
    
    return targets_pred, test_loss

def plot_learning_curve(history_dic, dataset, start_epoch=1, end_epoch=400, savefig=False):
    """
    Plot the learning curve for training and validation losses.

    This function visualizes the progression of the training and validation losses
    across epochs in logarithmic scale, providing insights into the model's performance
    during training.

    :param history_dic: Dictionary containing training history with keys 'loss' and optionally 'val_loss'.
                        Typically, this is `history.history` returned from `model.fit()`.
    :type history_dic: dict
    :param start_epoch: The epoch number at which the plot should start. Defaults to 1.
    :type start_epoch: int
    :param end_epoch: The epoch number at which the plot should end. Defaults to 400.
    :type end_epoch: int
    :param savefig: Whether to save the generated plot as a file. Defaults to False.
    :type savefig: bool
    
    :return: None
    """

    plt.plot(range(start_epoch, end_epoch), history_dic['loss'][start_epoch:])
    plt.plot(range(start_epoch, end_epoch), history_dic['val_loss'][start_epoch:])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title(f'Keras learning curve on {dataset} Monk problem')
    plt.legend(['loss TR','loss VL'])
    if savefig:
        plt.savefig(f"plot\keras{dataset}_learning", transparent = True)
    plt.show()

def plot_acc_curve(history_dic, dataset, start_epoch=1, end_epoch=400, savefig=False):
    """
    Plot the accuracy curve for training and validation sets.

    :param history_dic: Dictionary containing the training history. Keys should include 'accuracy' and optionally 'val_accuracy'.
    :type history_dic: dict
    :param start_epoch: Starting epoch for the plot. Defaults to 1.
    :type start_epoch: int
    :param end_epoch: Ending epoch for the plot. Defaults to 400.
    :type end_epoch: int
    :param savefig: Whether to save the plot as a file. Defaults to False.
    :type savefig: bool
    """

    plt.plot(range(start_epoch, end_epoch), history_dic['accuracy'][start_epoch:])
    plt.plot(range(start_epoch, end_epoch), history_dic['val_accuracy'][start_epoch:])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.yscale('log')
    plt.title(f'Keras accuracy on {dataset} Monk problem')
    plt.legend(['accuracy DV', 'accuracy TS'])
    if savefig:
        plt.savefig(f"plot\keras{dataset}_acc", transparent = True)
    plt.show()

def keras_network(ms = False, n_splits = 5, epochs = 140, dataset = 2):
    """
    Train and evaluate a neural network model on one of the Monk datasets using Keras, with k-fold cross-validation.

    :param ms: optional (default = False): If True, hyperparameters are selected using GridSearch.
    :type ms: bool
    :param n_splits: optional (default = 5): Number of splits for k-fold cross-validation.
    :type n_splits: int
    :param epochs: optional (default = 140): Number of epochs to train the model.
    :type epochs: int
    :param dataset: optional (default = 2): The dataset to use (1, 2, 3 for individual Monk datasets, or 'all' for all datasets).
    :type dataset: int or str

    :return: None
    :rtype: None
    """
    logger.info("Initializing Keras...")

    encoder = OneHotEncoder(sparse_output=False) 
    scaler = StandardScaler()

    if dataset == 1 or dataset == 2 or dataset ==3:
        iterations = 1
        logger.info(f"Monk {dataset} dataset has been selected...")

    elif dataset == 'all':
        iterations = 3
        logger.info("All the Monk datasets have been selected")

    else:
        logger.error("Invalid argument for dataset")

    for i in range(iterations): 
        data_path_train = abs_path(f'monks-{i+1}.train', 'data')
        data_path_test = abs_path(f'monks-{i+1}.test', 'data')

        if dataset == 1 or dataset == 2 or dataset ==3: 
            data_path_train = abs_path(f'monks-{dataset}.train', 'data')
            data_path_test = abs_path(f'monks-{dataset}.test', 'data') 
            
        # Reading and splitting the data
        features, targets = monk_data(data_path_train)
        features = encoder.fit_transform(features)

        features_test, targets_test = monk_data(data_path_test)
        features_test = encoder.transform(features_test)

        # Defining of the k-folding
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Model selection or manual adjustments
        if ms:
            logger.info("Choosing hyperparameters with a GridSearch")
            params = model_selection(features, targets, n_splits=n_splits, epochs=epochs)
        else:
            params = dict(model__eta=0.6, model__lmb=0.0001, model__alpha=0.5, model__batch_size=32)
            logger.info(f"Parameters have been chosen manually: {params}")

        # Creation of the model
        model = create_nn(input_shape = features.shape[1],
            eta = params["model__eta"],
            alpha = params["model__alpha"],
            lmb = params["model__lmb"],
            summary = True)

        prediction_model = create_nn(input_shape = features.shape[1],
            eta = params["model__eta"],
            alpha = params["model__alpha"],
            lmb = params["model__lmb"],
            summary = True)

        initial_weights = model.get_weights()

        best_model = None
        history = None
        mse_scores = []
        mse_best = float('inf')

        # Convert y_train to numpy array
        targets = np.array(targets, dtype=np.float32)
        targets_test = np.array(targets_test, dtype=np.float32)
        print("SHAPES before split:", features.shape, targets.shape)

        for train_index, val_index in folds.split(features, targets):
            features_train = features[train_index]
            features_val = features[val_index]
            targets_train = targets[train_index]
            targets_val = targets[val_index]

            model.set_weights(initial_weights)
            fit = model.fit(features_train, targets_train, epochs=epochs,
                        validation_data=(features_val, targets_val), verbose=0)

            targets_pred_val = model.predict(features_val)
            mse = mean_squared_error(targets_val, targets_pred_val)
            mse_scores.append(mse)

            if mse < mse_best:
                mse_best = mse
                best_model = model
                history = fit

        tr_loss = history.history['loss']
        val_loss = history.history['val_loss']

        prediction_model.set_weights(initial_weights)
        fit = prediction_model.fit(features, targets, epochs=epochs,
                                   validation_data=(features_test, targets_test), verbose=0)

        dv_accuracy = fit.history['accuracy']
        ts_accuracy = fit.history['val_accuracy']

        # Predicting on test set
        targets_pred, test_loss = predict(model=prediction_model, 
                                                            features_test=features_test, 
                                                            targets_test=targets_test)

        targets_test = targets_test.astype(int)
        targets_pred = (targets_pred > 0.5).astype(int)

        accuracy_metric = metrics.Accuracy()
        accuracy_metric.update_state(targets_test, targets_pred)
        test_accuracy = accuracy_metric.result().numpy()

        # Printing the losses and accuracy 
        print("TR loss (best-performing fold): ", tr_loss[-1])
        print("VL loss (best-performing fold): ", val_loss[-1])

        print("DV accuracy (training on both TR and VL): ", dv_accuracy[-1])
        print("TS accuracy from history (training on both TR and VL): ", ts_accuracy[-1])

        print("TS loss from prediction (training on both TR and VL): ", tf.reduce_mean(test_loss).numpy())
        print("TS accuracy from prediction (training on both TR and VL): ", tf.reduce_mean(test_accuracy).numpy())

        # Classification report on test set
        print("\nDetailed Classification Report:")
        print(classification_report(targets_test, targets_pred))

        logger.info("Computation with Keras successfully ended!")

        plot_learning_curve(history_dic=history.history, dataset = dataset, end_epoch=epochs, savefig=True)
        plot_acc_curve(history_dic=fit.history, dataset = dataset, end_epoch=epochs, savefig=True)

if __name__ == '__main__':
    keras_network(ms= True)