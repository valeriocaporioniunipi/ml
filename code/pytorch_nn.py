
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
from torch import optim
import matplotlib.pyplot as plt

from utils import torch_mee, abs_path, get_data, get_outer, w_csv

ms_result = []


class NeuralNetwork(nn.Module):
    def __init__(self, units=32, input=12, output=3, hidden_layers=3):
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

        logger.info(f"Model Architecture: {self}")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def init_weights(m):
    logger.info("Weights Glorot initialized")
    if type(m) == Linear:
        xavier_normal_(m.weight)

def plot_learning_curve(losses, val_losses, start_epoch=1, savefig=False, **kwargs):
    """
    function that shows the learning curve
    """
    plt.plot(range(start_epoch, kwargs['epochs']), losses[start_epoch:])
    plt.plot(range(start_epoch, kwargs['epochs']), val_losses[start_epoch:])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['loss TR', 'loss VL'])
    plt.title(f'PyTorch Learning Curve \n {kwargs}')

    if savefig:
        plt.savefig("NN_Torch.pdf", **kwargs)

    plt.show()

def make_train_step(model, loss_fn, optimizer):

    # Builds the inner function, which performs a step in the training loop
    def train_step(x, y):

        # setting the model to TRAIN mode
        model.train()
        # making predictions
        y_hat = model(x)
        # computing the loss
        loss = loss_fn(y, y_hat)
        # computing gradients
        loss.backward()
        # updating parameter and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # then the inner function returns the loss
        return loss.item()

    # the outer function returns the inner function, which will be called inside the training loop
    return train_step

def fit(x, y, model, optimizer, validation_data, loss_fn=torch_mee, epochs=200, batch_size=64):
    
    """"
    Fitting the model
    """
    # create the train_step function for our model, loss function and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    val_losses = []

    # change our data into tensors to work with PyTorch
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    x_val, y_val = validation_data
    x_val_tensor = torch.from_numpy(x_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    validation_data = TensorDataset(x_val_tensor, y_val_tensor)
    train_data = dataset

    # divide the dataset in batches
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        epoch_losses = []
        #updating weights batch by batch
        for x_batch, y_batch in train_loader:
            # perform one train step and return the corresponding loss for each epoch
            loss = train_step(x_batch, y_batch)
            epoch_losses.append(loss)
        losses.append(np.mean(epoch_losses))

        epoch_val_losses = []

        # disabling gradient update
        with torch.no_grad():

            for x_val, y_val in val_loader:
                # set model to VALIDATION mode
                model.eval()
                # makes predictions
                y_hat = model(x_val)
                # computes loss
                val_loss = loss_fn(y_val, y_hat)
                epoch_val_losses.append(val_loss.item())

            val_losses.append(np.mean(epoch_val_losses))
    return losses, val_losses


def cross_validation(x, y, n_splits, epochs, eta=0.003, alpha=0.85, lmb=0.0002, batch_size=64):
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=False)
    cv_loss = []
    fit_times = []
    fold_idx = 1


    # for each fold, whose number is defined by n_splits, create and fit a different model
    for tr_idx, vl_idx in kf.split(x, y):
        model = NeuralNetwork()
        model.apply(init_weights)
        optimizer = optim.SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lmb)

        fit_time = time.time()
        loss_tr, loss_vl = fit(x[tr_idx], y[tr_idx], model=model, optimizer=optimizer, epochs=epochs,
                               batch_size=batch_size, validation_data=(x[vl_idx], y[vl_idx]))

        fit_time = time.time() - fit_time
        fit_times.append(fit_time)

        # results
        cv_loss.append([loss_tr[-1], loss_vl[-1]])


        fold_idx += 1

    params = dict(eta=eta, alpha=alpha, lmb=lmb, epochs=epochs, batch_size=batch_size)

    # calculate total time to make the entire cross validation process
    fit_time = np.sum(fit_times)
    return params, cv_loss, fit_time


# callback function for the multiprocessing task
def log_ms_result(result):
    ms_result.append(result)


def model_selection(x, y, n_splits, epochs):
    # define a pool of tasks, for multiprocessing
    pool = mp.Pool(processes=mp.cpu_count())

    # grid search parameters
    #eta = np.arange(start=0.003, stop=0.01, step=0.001)
    eta = [0.005, 0.05, 0.5]
    eta = [float(round(i, 4)) for i in list(eta)]

    #alpha = np.arange(start=0.4, stop=1, step=0.1)
    alpha = [0.2, 0.4, 0.6, 0.8]
    alpha = [float(round(i, 1)) for i in list(alpha)]

    #lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)
    lmb = [0.00005, 0.0001, 0.001, 0.01]
    lmb = [float(round(i, 5)) for i in list(lmb)]

    batch_size = [32, 64, 128]

    param_grid = dict(eta=eta, alpha=alpha, lmb=lmb, batch_size = batch_size)

    # computing the grid size
    grid_size =  len(eta) * len(alpha) * len(lmb) * len(batch_size)

    ms__initial_time = time.time()
    print(f"Starting grid search: {grid_size} fits")

    param_combinations = list(itertools.product(*param_grid.values()))

    for params in param_combinations:
        # performing cross validation via multiprocessing
        pool.apply_async(cross_validation, args = (x, y, n_splits, epochs, *params),
                        callback=log_ms_result)
    pool.close()
    pool.join()

    ms_elapsed_time = time.time()-ms__initial_time
    logger.info(f"Grid search successfully performed in {ms_elapsed_time} seconds")


    # print model selection results
    sorted_res = sorted(ms_result, key=lambda tup: (np.mean(tup[1], axis=0))[1])
    for (p, l, t) in sorted_res:
        scores = np.mean(l, axis=0)
        print("{} \t TR {:.4f} \t TS {:.4f} (Fit Time: {:.4f})".format(p, scores[0], scores[1], t))

    min_loss = (np.mean(sorted_res[0][1], axis=0))[1]
    best_params = sorted_res[0][0]

    print("\nBest score {:.4f} with {}\n".format(min_loss, best_params))
    return best_params


def predict(model, x_test, y_test, x_outer):
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


def pytorch_nn(ms=False, n_splits=10 , epochs =200):
    logger.info("Initializing PyTorch...")

    filepath = abs_path("ML-CUP24-TR.csv", "data")
    # extracting features and targets from csv
    features, targets, features_test, targets_test = get_data(filepath, split = True)

    # choose model selection or hand-given parameters
    if ms:
        logger.info("Choosing hyperparameters with a GridSearch")
        params = model_selection(features, targets, n_splits = n_splits, epochs = epochs)
    else:
        params = dict(eta=0.003, alpha=0.85, lmb=0.0002, epochs=80, batch_size=64)
        logger.info(f"Parameters have been chosen manually: {params}")

    # create and fit the model
    model = NeuralNetwork()
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=params['eta'],
                    momentum=params['alpha'], weight_decay=params['lmb'])

    tr_losses, val_losses = fit(features, targets, model=model, optimizer=optimizer,
                                batch_size=params['batch_size'], epochs=params['epochs'])

    y_pred_outer, internal_losses = predict(model=model,
                                x_outer = get_outer(abs_path("ML-CUP24-TS.csv", "data")),
                                x_test= features_test,
                                y_test = targets_test)

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(internal_losses))

    logger.info("Computation with PyTorch successfully ended!")

    plot_learning_curve(tr_losses, val_losses, savefig=True, **params)

    # generate csv file for MLCUP
    w_csv(y_pred_outer)

if __name__ == '__main__':
    pytorch_nn()