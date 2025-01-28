
import time
import itertools
import multiprocessing as mp

from loguru import logger
import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch import optim
from utils import torch_mee, abs_path, get_data, get_outer, w_csv, standardize_data

ms_result = []
np.random.seed(42)

class NeuralNetwork(nn.Module):
    """
    A neural network class built using PyTorch's nn.Module to solve a multi-class classification problem.

    :param units: optional (default = 32): number of units in each hidden layer
    :type units: int
    :param input: optional (default = 12): number of input features
    :type input: int
    :param output: optional (default = 3): number of output classes
    :type output: int
    :param hidden_layers: optional (default = 3): number of hidden layers
    :type hidden_layers: int

    :return: None
    :rtype: None
    """
    def __init__(self, units=32, input=12, output=3, hidden_layers=3):
        """
        Initializes the neural network with specified number of units in hidden layers, input size, and output size.

        :param units: optional (default = 32): number of units in each hidden layer
        :param input: optional (default = 12): number of input features
        :param output: optional (default = 3): number of output classes
        :param hidden_layers: optional (default = 3): number of hidden layers
        """
        super().__init__()
        self.flatten = nn.Flatten()

        # Input layer
        layers = [nn.Linear(input, units), nn.ReLU()]

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(units, units))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(units, output))

        self.linear_relu_stack = nn.Sequential(*layers)

        logger.info(f"Model architecture: {self}")

    def forward(self, features):
        """
        Defines the forward pass for the neural network.

        :param features: input features to the network (Tensor)
        :type features: torch.Tensor

        :return: logits (raw output of the network before activation)
        :rtype: torch.Tensor
        """
        features = self.flatten(features)
        logits = self.linear_relu_stack(features)
        return logits

def init_weights(m):
    """
    Initializes the weights of a linear layer using the Xavier normal distribution.

    :param m: the module (layer) whose weights are to be initialized
    :type m: nn.Module

    :return: None
    :rtype: None
    """
    if type(m) == Linear:
        xavier_normal_(m.weight)

def plot_learning_curve(losses, val_losses, epochs, start_epoch=1, savefig=False):
    """
    Plots the learning curve showing the training loss and validation loss over epochs.

    :param losses: list of training losses for each epoch
    :type losses: list
    :param val_losses: list of validation losses for each epoch
    :type val_losses: list
    :param epochs: total number of epochs for the plot
    :type epochs: int
    :param start_epoch: optional (default = 1): the starting epoch for the plot
    :type start_epoch: int
    :param savefig: optional (default = False): whether to save the plot as a PDF file
    :type savefig: bool

    :return: None
    :rtype: None
    """
    plt.plot(range(start_epoch, epochs), losses[start_epoch:])
    plt.plot(range(start_epoch, epochs), val_losses[start_epoch:], ls = 'dashed')

    plt.xlabel("epoch", fontsize = 20)
    plt.ylabel("loss", fontsize = 20)
    plt.legend(['loss TR', 'loss VL'], fontsize = 20)
    plt.title(f'PyTorch learning curve', fontsize = 20)
    #plt.yscale('log')

    if savefig:
        plt.savefig("plot/NN_Torch.pdf", transparent = True)

    plt.show()

def make_train_step(model, loss_fn, optimizer):
    """
    Creates a training step function that performs one iteration of training:
    setting the model to training mode, making predictions, computing loss,
    performing backpropagation, updating model parameters, and zeroing gradients.

    :param model: the neural network model
    :type model: nn.Module
    :param loss_fn: the loss function used for calculating the error
    :type loss_fn: callable
    :param optimizer: the optimizer used to update the model parameters
    :type optimizer: Optimizer

    :return: A function that performs one training step
    :rtype: function
    """

    # Builds the inner function, which performs a step in the training loop
    def train_step(features, targets):
        """
        Inner function of make_train_step
        """

        # setting the model to TRAIN mode
        model.train()
        # making predictions
        y_hat = model(features)
        # computing the loss
        loss = loss_fn(y_hat, targets)
        # computing gradients
        loss.backward()
        # updating parameter and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # then the inner function returns the loss
        return loss.item()

    # the outer function returns the inner function, which will be called inside the training loop
    return train_step


def fit(x_train, y_train, model, optimizer, validation_data = None, loss_fn=torch_mee, epochs=200, batch_size=64):
    """
    Trains the model for a specified number of epochs, updating weights using backpropagation, and computes the loss
    for both training and validation datasets. Optionally, applies learning rate scheduling.

    :param x_train: training features
    :type x_train: numpy.ndarray
    :param y_train: training labels
    :type y_train: numpy.ndarray
    :param model: the neural network model
    :type model: nn.Module
    :param optimizer: the optimizer used to update the model parameters
    :type optimizer: Optimizer
    :param validation_data: optional (default = None): a tuple of validation features and labels
    :type validation_data: tuple, optional
    :param loss_fn: optional (default = torch_mee): the loss function used for calculating the error
    :type loss_fn: callable
    :param epochs: optional (default = 200): the number of training epochs
    :type epochs: int
    :param batch_size: optional (default = 64): the batch size used for training
    :type batch_size: int

    :return: training losses (and validation losses if validation data is provided)
    :rtype: list (or tuple of lists)
    """

    # define a scheduler for variable eta (decaying learning rate)
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = 1, end_factor = 1e-6, total_iters=400)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # create the train_step function for our model, loss function and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    val_losses = []

    # change the data into tensors to work with PyTorch
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).float()

    train_data = TensorDataset(x_tensor, y_tensor)
    # divide the dataset in batches
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    if validation_data != None:

        x_val, y_val = validation_data[0], validation_data[1]
        x_val_tensor = torch.from_numpy(x_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()

        validation_data = TensorDataset(x_val_tensor, y_val_tensor)
        # divide the dataset in batches
        val_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)

    
    for _ in range(epochs):
        epoch_losses = []
        #updating weights batch by batch
        for x_batch, y_batch in train_loader:
            # performing one train step and returning the loss for each epoch
            loss = train_step(x_batch, y_batch)
            epoch_losses.append(loss)
        scheduler.step(metrics = loss)
        losses.append(np.mean(epoch_losses))

        epoch_val_losses = []
        
        if validation_data != None:
            # disabling gradient update
            with torch.no_grad():

                for x_val, y_val in val_loader:
                    # set model to VALIDATION mode
                    model.eval()
                    # make predictions
                    y_hat = model(x_val)
                    # compute loss
                    val_loss = loss_fn(y_val, y_hat)
                    epoch_val_losses.append(val_loss.item())

            val_losses.append(np.mean(epoch_val_losses))
            
    if validation_data == None:
        return losses
    else:       
        return losses, val_losses


def cross_validation(features, targets,
                    n_splits, epochs,
                    eta, alpha, lmb, batch_size):
    """
    Performs k-fold cross-validation by training and evaluating the model on different splits of the data.
    This function trains a model for each fold and returns the training and validation losses, as well as the total fitting time.

    :param features: the input features for training
    :type features: numpy.ndarray
    :param targets: the target labels for training
    :type targets: numpy.ndarray
    :param n_splits: the number of splits for cross-validation (k-fold)
    :type n_splits: int
    :param epochs: number of epochs to train the model
    :type epochs: int
    :param eta: learning rate for the optimizer
    :type eta: float
    :param alpha: momentum for the optimizer
    :type alpha: float
    :param lmb: regularization parameter (L2)
    :type lmb: float
    :param batch_size: the batch size for training
    :type batch_size: int

    :return: A tuple containing:
        - params: a dictionary with the hyperparameters used
        - cv_loss: a list of lists containing training and validation losses for each fold
        - fit_time: the total fitting time for the cross-validation process
    :rtype: tuple
    """
    
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    cv_loss = []
    fit_times = []

    # for each fold, whose number is defined by n_splits, create and fit a different model
    for i, (tr_idx, vl_idx) in enumerate(kf.split(features), 1):
        model = NeuralNetwork()
        model.apply(init_weights)
        optimizer = optim.SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lmb)

        fit_time = time.time()
        loss_tr, loss_vl = fit(features[tr_idx], targets[tr_idx], model=model, optimizer=optimizer, epochs=epochs,
                               batch_size=batch_size, validation_data=(features[vl_idx], targets[vl_idx]))

        fit_time = time.time() - fit_time
        fit_times.append(fit_time)

        # results
        cv_loss.append([loss_tr[-1], loss_vl[-1]])

    params = dict(eta=eta, alpha=alpha, lmb=lmb, epochs=epochs, batch_size=batch_size)

    # calculate total time to make the entire cross validation process
    fit_time = np.sum(fit_times)
    return params, cv_loss, fit_time


# callback function for the multiprocessing task
def log_ms_result(result):
    """
    Logs the result of a model selection (MS) process by appending the result to a list.

    :param result: the result of the model selection process
    :type result: any type (e.g., float, dict, etc.)

    :return: None
    """
    ms_result.append(result)


def model_selection(features, targets, n_splits, epochs):
    """
    Performs a grid search for hyperparameter tuning using cross-validation and multiprocessing.
    
    This function conducts a grid search over the specified hyperparameters (`eta`, `alpha`, `lmb`, 
    and `batch_size`) for model training. It uses the `cross_validation` function and logs the results 
    of each parameter combination.

    :param features: the input features for training the model
    :type features: numpy.ndarray
    :param targets: the target labels for training the model
    :type targets: numpy.ndarray
    :param n_splits: the number of splits for cross-validation
    :type n_splits: int
    :param epochs: the number of epochs for training the model
    :type epochs: int

    :return: the hyperparameters corresponding to the best performing model
    :rtype: dict
    """
    # define a pool of tasks, for multiprocessing
    pool = mp.Pool(processes=mp.cpu_count())

    # grid search parameters
    #eta = np.arange(start=0.003, stop=0.01, step=0.001)
    eta = [0.0005, 0.005, 0.05, 0.5]
    eta = [0.002, 0.005, 0.007]
    eta = [float(round(i, 4)) for i in list(eta)]

    #alpha = np.arange(start=0.4, stop=1, step=0.1)
    alpha = [0.6, 0.8, 1]
    alpha = [0.7, 0.8, 0.9]
    alpha = [float(round(i, 1)) for i in list(alpha)]

    #lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)
    lmb = [0.00005, 0.0001, 0.001]
    lmb = [0.00007, 0.0001, 0.0005]
    lmb = [float(round(i, 5)) for i in list(lmb)]

    batch_size = [64, 128, 250]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb, batch_size = batch_size)

    # computing the grid size
    grid_size =  len(eta) * len(alpha) * len(lmb) * len(batch_size)

    ms__initial_time = time.time()
    logger.info(f"Starting grid search: {grid_size} fits")

    param_combinations = list(itertools.product(*param_grid.values()))

    for params in param_combinations:
        # performing cross validation via multiprocessing
        pool.apply_async(cross_validation, args = (features, targets, n_splits, epochs,
                                                 params[0], params[1], params[2], params[3]),
                          callback= log_ms_result)
        # ms_result is the callback of the pool.apply_async: it will return a ms_result
    pool.close() # closing the pool
    pool.join()

    ms_elapsed_time = time.time() - ms__initial_time
    logger.info(f"Grid search successfully performed in {ms_elapsed_time} seconds")


    # print model selection results

    sorted_res = sorted(ms_result, key=lambda tup: (np.mean(tup[1], axis=0))[1])
    for (p, l, t) in sorted_res:
        scores = np.mean(l, axis=0)
        #print("{} \t TR {:.4f} \t TS {:.4f} (Fit Time: {:.4f})".format(p, scores[0], scores[1], t))

    min_loss = (np.mean(sorted_res[0][1], axis=0))[1]
    best_params = sorted_res[0][0]

    print("\nBest score {:.4f} with {}\n".format(min_loss, best_params))
    return best_params


def predict(model, x_test, y_test, x_outer):
    """
    Makes predictions using the trained model on both an internal test set and an external (blind) test set.

    :param model: the trained model used to make predictions
    :type model: nn.Module
    :param x_test: the input features of the internal test set
    :type x_test: numpy.ndarray
    :param y_test: the true target labels of the internal test set
    :type y_test: numpy.ndarray
    :param x_outer: the input features of the external (blind) test set
    :type x_outer: numpy.ndarray

    :return: predicted targets for the external test set and the loss on the internal test set
    :rtype: tuple (numpy.ndarray, float)
    """
    # change our data into tensors to work with PyTorch
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    x_outer = torch.from_numpy(x_outer).float()

    # predict on internal test set
    y_ipred = model(x_test)
    iloss = torch_mee(y_test, y_ipred)

    # predict on blind test set
    y_outer_pred = model(x_outer)

    # return predicted target on blind test set and losses on internal test set
    return y_outer_pred.detach().numpy(), iloss.item()


def pytorch_nn(ms=True, n_splits=10 , epochs =500):
    """
    Initializes and trains a PyTorch neural network model, with optional hyperparameter tuning via grid search.

    :param ms: optional (default=True), whether to perform model selection using grid search
    :type ms: bool
    :param n_splits: optional (default=10), number of splits for cross-validation
    :type n_splits: int
    :param epochs: optional (default=2000), number of training epochs
    :type epochs: int

    :return: None
    """
    logger.info("Initializing PyTorch...")

    filepath = abs_path("ML-CUP24-TR.csv", "data")
    # extracting features and targets from csv
    features, targets, features_test, targets_test = get_data(filepath, split = True)
        # Standardize features and targets
    features, features_test, targets, targets_test = standardize_data(features, features_test, targets, targets_test)
    # choose model selection or hand-given parameters
    if ms:
        logger.info("Choosing hyperparameters with a GridSearch")
        params = model_selection(features, targets, n_splits = n_splits, epochs = epochs)
    else:
        params = dict(eta=0.005, alpha=0.8, lmb=0.0001, epochs=epochs, batch_size=128)
        logger.info(f"Parameters have been chosen manually: {params}")

    # create and fit the model
    model = NeuralNetwork()
    pred_model = NeuralNetwork()
    model.apply(init_weights)
    pred_model.apply(init_weights)
    pred_optimizer = optim.SGD(pred_model.parameters(), lr=params['eta'],
                    momentum=params['alpha'], weight_decay=params['lmb'])
    optimizer = optim.SGD(model.parameters(), lr=params['eta'],
                    momentum=params['alpha'], weight_decay=params['lmb'])

    val_perc = (1/n_splits)
    val_size = int(val_perc * len(features))

    indices = np.random.permutation(len(features))

    # divide the indices into two groups
    indices_train = indices[val_size:]  
    indices_val = indices[:val_size]

    # divide data based on indices
    features_val = features[indices_val]
    features_train = features[indices_train]

    targets_val = targets[indices_val]
    targets_train = targets[indices_train]   

    tr_losses, val_losses = fit(features_train, targets_train, model=model, optimizer=optimizer, validation_data = (features_val, targets_val),
                                batch_size=params['batch_size'], epochs=params['epochs'])
    
    # initialize and fit the model on both TR and VL
    tr_pred_losses = fit(features, targets, model=pred_model, optimizer=pred_optimizer,
                 batch_size=params['batch_size'], epochs=params['epochs'])

    y_pred_outer, internal_losses = predict(model=pred_model,
                                x_outer = get_outer(abs_path("ML-CUP24-TS.csv", "data")),
                                x_test= features_test,
                                y_test = targets_test)

    print("TR loss (best-performing fold): ", tr_losses[-1])
    print("VL loss (best-performing fold): ", val_losses[-1])
    print("DV loss: ", tr_pred_losses[-1])
    print("TS loss (training on both TR and VL): ", internal_losses)
   
    logger.info("Computation with PyTorch successfully ended!")

    plot_learning_curve(tr_losses, val_losses, epochs=epochs, savefig=True)

    # generate csv file for MLCUP
    w_csv(y_pred_outer)

if __name__ == '__main__':
    pytorch_nn(ms= True)