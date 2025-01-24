
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
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch import optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

from utils import torch_mee, abs_path, monk_data

ms_result = []
np.random.seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17, 16),  # Adjusted to accept 17 features
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def init_weights(m):
    if type(m) == Linear:
        xavier_normal_(m.weight)

def plot_learning_curve(losses, val_losses, epochs, start_epoch=1, savefig=False):

    plt.plot(range(start_epoch, epochs), losses[start_epoch:])
    plt.plot(range(start_epoch, epochs), val_losses[start_epoch:])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['loss TR', 'loss VL'])
    plt.title(f'PyTorch learning curve')
    #plt.yscale('log')

    if savefig:
        plt.savefig("plot/monk_pytorch_1.pdf")
    plt.show()

def make_train_step(model, loss_fn, optimizer):

    # Builds the inner function, which performs a step in the training loop
    def train_step(features, targets):

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


def fit(x_train, y_train, model,
        optimizer, validation_data = None, loss_fn=torch_mee, epochs=200, batch_size=64):

    # define a scheduler for variable eta (decaying learning rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = 1,
                                            end_factor = 0.01, total_iters=400)
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
        scheduler.step()
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
    ms_result.append(result)


def model_selection(features, targets, n_splits, epochs):
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
        print("{} \t TR {:.4f} \t TS {:.4f} (Fit Time: {:.4f})".format(p, scores[0], scores[1], t))

    min_loss = (np.mean(sorted_res[0][1], axis=0))[1]
    best_params = sorted_res[0][0]

    print("\nBest score {:.4f} with {}\n".format(min_loss, best_params))
    return best_params


def predict(model, features, targets):
    predictions = model.predict(features)
    mee = torch_mee(targets, predictions)
    accuracy = accuracy_score(targets, predictions)
    return predictions, mee, accuracy


def monk_pytorch(ms=True, n_splits=10 , epochs =2000):

    # # Encoder
    # encoder = OneHotEncoder()

    # # Getting the path to the file
    # data_path_train = abs_path('monks-3.train', 'data')
    # data_path_test = abs_path('monks-3.test', 'data')

    # # Splitting and encoding the training data
    # features, targets = monk_data(data_path_train)
    # features = encoder.fit_transform(features)

    # # Splitting and encoding the test data
    # features_test, targets_test = monk_data(data_path_test)
    # features_test = encoder.transform(features_test)
    encoder = OneHotEncoder()
    features, targets = monk_data(abs_path('monks-3.train', 'data'))
    features = encoder.fit_transform(features).toarray()
    features_test, targets_test = monk_data(abs_path('monks-3.test', 'data'))
    features_test = encoder.transform(features_test).toarray()
    targets, targets_test = np.array(targets.values, dtype=np.float32), np.array(targets_test.values, dtype = np.float32)
    indices = np.random.permutation(features.shape[0])

    # choose model selection or hand-given parameters
    if ms:
        logger.info("Choosing hyperparameters with a GridSearch")
        params = model_selection(features, targets, n_splits = n_splits, epochs = epochs)
    else:
        params = dict(eta=0.002, alpha=0.5, lmb=0.005, epochs=epochs, batch_size=64)
        logger.info(f"Parameters have been chosen manually: {params}")

    # create and fit the model
    model = NeuralNetwork()
    pred_model = NeuralNetwork()
    model.apply(init_weights)
    pred_model.apply(init_weights)
    optimizer = optim.SGD(pred_model.parameters(), lr=params['eta'],
                    momentum=params['alpha'], weight_decay=params['lmb'])
    pred_optimizer = optim.SGD(model.parameters(), lr=params['eta'],
                    momentum=params['alpha'], weight_decay=params['lmb'])

    val_perc = (1/n_splits)
    val_size = int(val_perc * len(features))

    # indices = np.random.permutation(len(features))

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
    tr_pred_losses = fit(features, targets, model=pred_model, optimizer=pred_optimizer, validation_data=(features_test, targets_test),
                 batch_size=params['batch_size'], epochs=params['epochs'])
    
    development_accuracy= accuracy_score(targets, pred_model(torch.from_numpy(features).float()))
    pred_test, test_loss, test_accuracy = predict(model = pred_model, features = features, targets=targets) 
    print("TR loss (best-performing fold): ", tr_losses[-1])
    print("VL loss (best-performing fold): ", val_losses[-1])
    print("DV loss: ", tr_pred_losses[-1])

    print("Development accuracy: ", development_accuracy)
    print("Test accuracy: ", test_accuracy)
   
    logger.info("Computation with PyTorch successfully ended!")

    plot_learning_curve(tr_losses, val_losses, epochs=epochs, savefig=True)


if __name__ == '__main__':
    monk_pytorch(ms= False)