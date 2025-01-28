import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import monk_data, abs_path
from itertools import product
from loguru import logger


class BinaryClassificationNN(nn.Module):
    """
    Neural network architecture for binary classification.
    
    Implements a neural network for binary classification tasks.
    
    :param input_size: Number of input features
    :type input_size: int
    :param hidden_size: Number of neurons in hidden layers
    :type hidden_size: int
    """
    def __init__(self, input_size, hidden_size):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the neural network.
        
        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output predictions
        :rtype: torch.Tensor
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.activation(self.output(x))
        return x


def define_neural_network(input_size, hidden_size=16):
    """
    Defines the neural network architecture for binary classification.
    
    :param input_size: The size of the input features
    :type input_size: int 
    :param hidden_size: Number of hidden units in the layer (default is 32)
    :type hidden_size: int
    
    :return: A binary classification neural network
    :rtype: nn.Module
    """
    return BinaryClassificationNN(input_size, hidden_size)


def weights_init(m):
    """
    Initializes the model weights using a normal distribution.
    
    :param m: A PyTorch model layer
    :type m: nn.Module
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

def model_selection(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    """
    Performs grid search for hyperparameter optimization.
    
    :return: The best model parameters and best model
    :rtype: tuple(dict, BinaryClassificationNN)
    """
    # Adjusted hyperparameter ranges for better convergence
    etas = [0.1,0.01, 0.001]  # Learning rates
    lambdas = [1, 1e-1,1e-2,1e-3,1e-4]  # Weight decay values
    
    best_model = None
    best_val_loss = float('inf')
    best_params = None
    patience = 10  # Early stopping patience
    
    start_time = time.time()
    logger.info("Starting Grid Search for hyperparameter optimization")
    
    for eta, lmb in product(etas, lambdas):
        model = define_neural_network(X_train_tensor.shape[1])
        model.apply(weights_init)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=lmb)
        
        epochs = 300
        no_improve = 0
        min_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            with torch.no_grad():
                model.eval()
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
                # Check if this is the best model so far
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    break  # Early stopping
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {'eta': eta, 'lambda': lmb}
                    best_model = model.state_dict()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Grid search concluded in {elapsed_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Create a new model with the best parameters
    final_model = define_neural_network(X_train_tensor.shape[1])
    final_model.load_state_dict(best_model)
    
    return best_params, final_model

def train_final_model(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
                     X_test_tensor, y_test_tensor, best_model, best_params, epochs=300):
    """
    Trains the final model using the best hyperparameters found during grid search.
    
    :param best_model: Pre-trained model from grid search
    :type best_model: BinaryClassificationNN
    """
    best_model = define_neural_network(X_train_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        best_model.parameters(),
        lr=best_params['eta'],
        weight_decay=best_params['lambda']
    )
    best_model.apply(weights_init)
    
    # Training metrics storage
    train_losses = []
    val_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        best_model.train()
        optimizer.zero_grad()
        train_outputs = best_model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        with torch.no_grad():
            best_model.eval()
            # Calculate training metrics
            train_predicted = (train_outputs >= 0.5).float()
            train_accuracy = (train_predicted == y_train_tensor).float().mean()
            train_accuracies.append(train_accuracy.item())
            
            # Calculate validation metrics
            val_outputs = best_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
            
            
            # Calculate test metrics
            test_outputs = best_model(X_test_tensor)
            test_predicted = (test_outputs >= 0.5).float()
            test_accuracy = (test_predicted == y_test_tensor).float().mean()
            test_accuracies.append(test_accuracy.item())
            
            # Log progress periodically
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Test Acc: {test_accuracy:.4f}")
    
    return train_losses, val_losses, train_accuracies, test_accuracies



def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, test_accuracies):
    """
    Plots the loss and accuracy curves for the training process.
    
    :param train_losses: List of training loss values
    :type train_losses: list
    :param val_losses: List of validation loss values
    :type val_losses: list
    :param train_accuracies: List of training accuracy values
    :type train_accuracies: list
    :param test_accuracies: List of test accuracy values
    :type test_accuracies: list
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy', linestyle='--', color='orange')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def torch_main():
    """
    Main function to run the MONK dataset classification.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    # Load and preprocess data
    features_test, targets_test = monk_data(abs_path('monks-3.test', 'data'))
    features, targets = monk_data(abs_path('monks-3.train', 'data'))
    
    # Apply one-hot encoding to features
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(features)
    X_test_encoded = encoder.transform(features_test)
    
    # Split training data with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    # Convert to numpy arrays first, ensuring correct data types
    y_train = y_train.astype(float).to_numpy().reshape(-1, 1)
    y_val = y_val.astype(float).to_numpy().reshape(-1, 1)
    y_test = targets_test.astype(float).to_numpy().reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Train and evaluate model
    best_params, best_model = model_selection(X_train_tensor, y_train_tensor, 
                                            X_val_tensor, y_val_tensor)
    print(f'Best Params: {best_params}')
    
    metrics = train_final_model(X_train_tensor, y_train_tensor, X_val_tensor, 
                              y_val_tensor, X_test_tensor, y_test_tensor, 
                              best_model, best_params)
    
    plot_loss_and_accuracy(*metrics)


if __name__ == "__main__":
    torch_main()