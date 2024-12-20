import time
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
from keras import Sequential
from keras import layers
from keras import regularizers
from keras import optimizers
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

from utils import abs_path, get_data, euclidean_error, mean_euclidean_error, scorer 

# random seed 
seed = 42
np.random.seed(seed)

def create_nn(input_shape,
                  hidden_layers = 3,
                  hidden_nodes = 32,
                  dropout = 0.00,
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
    :param alpha: optional(default = 0.0001) regularization parameter
    :type alpha: float
    :param init_mode: optional(default = 'glorot_normal') kernel initializer
    :type init_mode: str
    
    :return: neural network model
    :rtype: Sequential
    """

    model = Sequential() # Defining the model
    model.add(layers.Input(shape=input_shape)) # Placing an input layer
    model.add(layers.Dropout(dropout)) # Placing dropout layer
    model.add(layers.BatchNormalization()) # BatchNormalization layer

    # Adding variable number of hidden layers (Dense+Dropout+BatchNormalization)
    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_nodes, activation=activation,
                                kernel_initializer=init_mode,
                                kernel_regularizer=regularizers.L2(lmb)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='linear'))  # Output layer of a regression problem

    # Adding optimizer for the model:
    optimizer = optimizers.SGD(learning_rate=eta, momentum=alpha)

    # Compiling the model
    model.compile(loss=euclidean_error, optimizer=optimizer, metrics=[euclidean_error])

    # Printing summary, if specified
    if summary:
        logger.info("Model successfully compiled, showing detailed summary ")
        model.summary()
    else: 
        pass
    return model

def model_selection(features, targets, n_splits, epochs):
    input_shape = np.shape(features[0])

    # grid search parameters
    eta = np.arange(start=0.003, stop=0.01, step=0.001)
    eta = [float(round(i, 4)) for i in list(eta)]

    alpha = np.arange(start=0.4, stop=1, step=0.1)
    alpha = [float(round(i, 1)) for i in list(alpha)]

    lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)
    lmb = [float(round(i, 4)) for i in list(lmb)]

    batch_size = [16, 32, 64]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb, batch_size=batch_size)


    model = KerasRegressor(model=lambda hidden_layers, hidden_nodes,dropout, eta, alpha, lmb:
        create_nn(input_shape, hidden_layers=hidden_layers, hidden_nodes=hidden_nodes,
        dropout=dropout, eta = eta, alpha = alpha, lmb = lmb), epochs=epochs, verbose=0)
    
    # k-folding definition
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, refit = False,
        cv = kf, n_jobs = -1, return_train_score=True, verbose = 1)
    
    # rescaling features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # fitting grid search
    start_time = time.time()
    logger.info("Starting Grid Search for hyperparameter optimization")
    grid_result = grid.fit(features_scaled, targets)

    # elapsed time
    end_time =  time.time() 
    elapsed_time = end_time- start_time
    logger.info(f"Grid search concluded {elapsed_time}")
    best_params = grid_result.best_params_
    # summarizing results
    logger.info(f"Best: {grid_result.best_score_} using {best_params}")
    return best_params

def predict(model, x_test, y_test, x_outer):
    # predict on internal test set of data
    y_pred = model.predict(x_test)
    loss = mean_euclidean_error(y_test, y_pred)
    # predict on an outer test set
    y_outer_pred = model.predict(x_outer)

    # return prediction on outer test set and loss on internal test set
    return y_outer_pred, loss

def plot_learning_curve(history, start_epoch=1, end_epoch = 200, savefig=False):

    lgd = ['loss TR']
    plt.plot(range(start_epoch, end_epoch), history['loss'][start_epoch:])
    if "val_loss" in history:
        plt.plot(range(start_epoch, end_epoch), history['val_loss'][start_epoch:])
        lgd.append('loss VL')

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title(f'Keras learning curve')
    plt.legend(lgd)

    if savefig:
        plt.savefig("NN_Keras.pdf", transparent = True)
    plt.show()


def keras_network(model_selection = True, n_splits=5, epochs = 200):
    logger.info("Initializing Keras...")
    # getting the absolute path to te file through utils function abs_path 
    filepath = abs_path("ML-CUP24-TR.csv", "data")
    # extracting features and targets from csv
    features, targets, features_test, targets_test = get_data(filepath)
    # Standardization of features
    scaler = StandardScaler()
    # definition of the k-folding
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # model selection or manual adjustments
    if model_selection:
        params = model_selection(features, targets, n_splits=n_splits, epochs=epochs)
    else:
        params = dict(eta=0.002, lmb=0.0001, alpha=0.7, epochs=200, batch_size=64)
    
    # the model is now created
    model = create_nn(input_shape = np.shape(features[0]),
        eta = params["eta"],
        alpha = params["model__alpha"],
        lmb = params["model__lmb"],
        summary = True)

    initial_weights = model.get_weights()
    best_model = None
    history = None
    mee_scores = []
    mee_best = float('inf')

    # Initialize plots for target values
    colormap = cmaps.get_cmap('tab20')
    colors = [colormap(i) for i in range(n_splits + 1)]
    # prepare 2x2 grid for 3 plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns grid
    ax = ax.flatten()  # flattening to make it easier to access the axes

    # leave the last subplot empty
    for i in range(3):  # loop for 3 targets
        ax[i].set_visible(True)  # make sure the axes are visible for the first 3 plots
        ax[i].set_xlabel('actual')
        ax[i].set_ylabel('predicted')
        ax[i].set_title(f'target {i+1} - actual vs predicted')
    # hide the last axis (empty subplot)
    ax[3].set_xlabel('loss')

    for i, (train_index, test_index) in enumerate(kf.split(features), 1):
        x_train, x_val = features[train_index], features[test_index]
        y_train, y_val = targets[train_index], targets[test_index]

        # Standardize features after the split
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model.set_weights(initial_weights)
        fit = model.fit(x_train, y_train, epochs=epochs, batch_size=params["batch_size"],
                            validation_data=(x_val, y_val), verbose=0)

        y_pred = model.predict(x_val)

        mee = mean_euclidean_error(y_val, y_pred)
        mee_scores.append(mee)

        if mee < mee_best:
            mee_best = mee
            best_model = model
            history = fit

        # Plotting actual vs predicted for each target dimension
        for j in range(3):  # Loop over each target dimension
            ax[j].scatter(y_val[:, j], y_pred[:, j], alpha=0.5, color=colors[i],
                        label=f'Fold {i} - Target {j+1} - MEE = {np.round(mee_scores[i-1], 2)}')
            ax[j].plot([y_val[:, j].min(), y_val[:, j].max()], 
                    [y_val[:, j].min(), y_val[:, j].max()], 'k--', lw=2)  # Ideal line y=x
            ax[j].legend()
        
        # fig.savefig('Keras_predictions.pdf', transparent=True)
        plot_learning_curve(history= history, savefig=True)





def training(features, targets, model, epochs, batch_size, n_splits = 5):
    """
    Train the neural network with k-fold cross-validation for multi-output regression.
    :param features: Feature matrix
    :type features: numpy.darray
    :param targets: 3D target array (n_samples, 3)
    :type targets: numpy.darray
    :param model: Keras model
    :type model: Sequential
    :param epochs: number of epochs to train the model
    :type epochs: int
    :param plot: set to True to show the "predicted vs actual" plots
    :type plot: Bool
    :param n_splits: optional (default = 5): n_splits for k-fold
    :type n_splits: int 

    :return: best model, mean absolute error, r2 score, residuals (prediction errors)
    """
    # Standardization of features
    scaler = StandardScaler()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    _, axh = plt.subplots(figsize=(10, 8))
    _, axp = plt.subplots(figsize=(10, 8))

    colormap = cmaps.get_cmap('tab20')
    colors = [colormap(i) for i in range(n_splits + 1)]

    initial_weights = model.get_weights()
    best_model = None
    mee_scores = []
    mee_best = float('inf')


    # prepare 2x2 grid for 3 plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns grid
    ax = ax.flatten()  # flattening to make it easier to access the axes

    # leave the last subplot empty
    for i in range(3):  # loop for 3 targets
        ax[i].set_visible(True)  # make sure the axes are visible for the first 3 plots
        ax[i].set_xlabel('Actual')
        ax[i].set_ylabel('Predicted')
        ax[i].set_title(f'Target {i+1} - Actual vs Predicted')
    # hide the last axis (empty subplot)
    ax[3].set_xlabel('loss')

    for i, (train_index, test_index) in enumerate(kf.split(features), 1):
        x_train, x_val = features[train_index], features[test_index]
        y_train, y_val = targets[train_index], targets[test_index]

        # Standardize features after the split
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model.set_weights(initial_weights)
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val), verbose=0)

        y_pred = model.predict(x_val)

        mee = mean_euclidean_error(y_val, y_pred)
        mee_scores.append(mee)

        if mee < mee_best:
            mee_best = mee
            best_model = model

        # Plotting actual vs predicted for each target dimension
        for j in range(3):  # Loop over each target dimension
            ax[j].scatter(y_val[:, j], y_pred[:, j], alpha=0.5, color=colors[i],
                        label=f'Fold {i} - Target {j+1} - MEE = {np.round(mee_scores[i-1], 2)}')
            ax[j].plot([y_val[:, j].min(), y_val[:, j].max()], 
                    [y_val[:, j].min(), y_val[:, j].max()], 'k--', lw=2)  # Ideal line y=x
            ax[j].legend()
    
    mee = np.mean(mee_scores)

        axh.set_xlabel("epoch")
        axh.set_ylabel("loss [log]")
        axh.set_title(f"TL over {epochs} epochs")
        axh.set_yscale('log')
        axh.legend()
        target_range = [targets.min(), targets.max()]
        axp.plot(target_range, target_range, 'k--', lw=2)
        axp.set_xlabel('data')
        axp.set_ylabel('prediction')
        axp.set_title('Actual vs predicted values')
        axp.legend(loc='upper left')
        axp.grid(False)

    return best_model, mee, history











