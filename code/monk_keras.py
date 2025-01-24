import time
import numpy as np
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
from keras import Sequential
from keras import optimizers
from keras import metrics
from keras import layers
from keras import regularizers

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
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
    Create a neural network model using Keras API in order to solve a regression problem.

    :param input_shape: shape of the data given to the input layer of the NN
    :type input_shape: tuple
    :param hidden_layers: optional(default = 3): number of hidden layers in the network
    :type hidden_layers: int
    :param hidden_nodes: optional(default = 32) number of nodes in each hidden layer
    :type hidden_nodes: int
    :param dropout: optional (default = 0.00): dropout rate of dropout layers
    :type dropout: float
    :param summary: optional (default = False): show the summary of the model
    :type summary: bool
    :param activation: optional(default = 'relu') activation function to use
    :type activation: str
    :param eta: optional(default = 0.002) learning rate of the SGD
    :type eta: float
    :param alpha: optional(default = 0.7) momentum of SGD
    :type alpha: float
    :param lmb: optional(default = 0.0001) regularization parameter
    :type lmb: float
    :type init_mode: str
    
    :return: neural network model
    :rtype: Sequential
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
    
    input_shape = np.shape(x)[1]
    
    model = KerasClassifier(model=create_nn, input_shape = input_shape, epochs=epochs, batch_size= 25, verbose=0)

    # Setting the grid search parameters
    eta = [0.05, 0.5]

    alpha = [0.4, 0.6]

    lmb = [0.001, 0.01]

    param_grid = dict(model__eta=eta, model__alpha=alpha, model__lmb=lmb)

    # K-folding definition
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', refit = 'accuracy',
        cv = folds, n_jobs = 1, return_train_score=True, verbose = 0)

    # Fitting grid search
    start_time = time.time()
    logger.info("Starting Grid Search for hyperparameter optimization")
    grid_result = grid.fit(x, y)

    end_time =  time.time() 
    elapsed_time = end_time - start_time
    logger.info(f"Grid search concluded {elapsed_time}")
    best_params = grid_result.best_params_

    # Summarizing results
    logger.info(f"Best: {grid_result.best_score_} using {best_params}")
    
    return best_params

def predict(model, features_test, targets_test):
    """
    Predicts the test values and computes the metrics

    :param model: model used for predictions
    :param features_test: features to classify
    :param targets_test: targets
    
    :return: 
    targets_pred, predictions on test set
    [test_loss], mean squared error on test set
    [test_accuracy], accuracy on test set
    
    """
    # Prediction
    targets_pred = model.predict(features_test)

    # Reshape targets_pred per match con targets_test
    targets_pred = targets_pred.flatten()  
    
    # Computing the mean_squared_error
    test_loss = mean_squared_error(targets_test, targets_pred)
    
    return targets_pred, test_loss

def plot_learning_curve(history_dic, dataset, start_epoch=1, end_epoch=400, savefig=False):

    lgd = ['loss TR']
    plt.plot(range(start_epoch, end_epoch), history_dic['loss'][start_epoch:])
    if "val_loss" in history_dic:
        plt.plot(range(start_epoch, end_epoch), history_dic['val_loss'][start_epoch:])
        lgd.append('loss VL')

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title(f'Keras learning curve')
    plt.legend(lgd)

    if savefig:
        plt.savefig(f"plot\keras{dataset}_mse", transparent = True)
    plt.show()

    lgd = ['accuracy TR']
    plt.plot(range(start_epoch, end_epoch), history_dic['accuracy'][start_epoch:])
    if "val_accuracy" in history_dic:
        plt.plot(range(start_epoch, end_epoch), history_dic['val_accuracy'][start_epoch:])
        lgd.append('accuracy VL')

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.yscale('log')
    plt.title(f'Keras learning curve')
    plt.legend(lgd)

    if savefig:
        plt.savefig(f"plot\keras{dataset}_acc", transparent = True)
    plt.show()

def keras_network(ms = False, n_splits = 5, epochs = 140, dataset = 1):
    logger.info("Initializing Keras...")

    encoder = OneHotEncoder(sparse_output=False) 

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
        fit = prediction_model.fit(features, targets, epochs=epochs, verbose=0)

        dv_accuracy = fit.history['accuracy']
        ts_accuracy = fit.history['accuracy']

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
        print("TS accuracy (training on both TR and VL): ", ts_accuracy[-1])

        print("TS loss (training on both TR and VL): ", tf.reduce_mean(test_loss).numpy())
        print("TS accuracy (training on both TR and VL): ", tf.reduce_mean(test_accuracy).numpy())

        # Classification report on test set
        print("\nDetailed Classification Report:")
        print(classification_report(targets_test, targets_pred))

        logger.info("Computation with Keras successfully ended!")

        plot_learning_curve(history_dic=fit.history, dataset = dataset, end_epoch=epochs, savefig=False)

if __name__ == '__main__':
    keras_network()