import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import monk_data, abs_path
from itertools import product
from loguru import logger

# Load MONK dataset
features_test, targets_test = monk_data(abs_path('monks-1.test', 'data'))
features, targets = monk_data(abs_path('monks-1.train', 'data'))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_test = scaler.transform(features_test)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, targets, test_size=0.2, random_state=42
)

# Convert labels to float (ensure they are 0 or 1)
y_train = y_train.astype(float).values.reshape(-1, 1)
y_val = y_val.astype(float).values.reshape(-1, 1)
y_test = targets_test.astype(float).values.reshape(-1, 1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network class
class BinaryClassificationNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassificationNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.activation(self.output(x))
        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.zeros_(m.bias)

# Grid search parameters
etas = [0.9, 0.8, 0.7, 0.6]
lambdas = [0, 0.001, 0.0001, 0.1, 0.01]
alphas = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
best_model = None
best_accuracy = 0

start_time = time.time()
logger.info("Starting Grid Search for hyperparameter optimization")

for eta, lmb, alpha in product(etas, lambdas, alphas):
    model = BinaryClassificationNN(X_train.shape[1], 16)
    model.apply(weights_init)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lmb)

    
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        model.eval()
        val_outputs = model(X_test_tensor)
        val_predicted = (val_outputs >= 0.5).float()
        val_accuracy = (val_predicted == y_test_tensor).float().mean().item()

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = {'eta': eta, 'lambda': lmb, 'alpha': alpha}

end_time = time.time() 
elapsed_time = end_time - start_time
logger.info(f"Grid search concluded in {elapsed_time}")
print(f'Best Params: {best_params}, Best Validation Accuracy: {best_accuracy:.4f}')

best_model = BinaryClassificationNN(X_train.shape[1], 16)
best_model.apply(weights_init)

# Final training with best params
best_optimizer = optim.SGD(best_model.parameters(), lr=best_params['eta'], weight_decay=best_params['lambda'], momentum=best_params['alpha'])
criterion = nn.MSELoss()
train_losses = []
val_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    best_model.train()
    best_optimizer.zero_grad()
    train_outputs = best_model(X_train_tensor)
    train_loss = criterion(train_outputs, y_train_tensor)
    train_loss.backward()
    best_optimizer.step()
    
    train_losses.append(train_loss.item())
    with torch.no_grad():
        train_predicted = (train_outputs >= 0.5).float()
        train_accuracy = (train_predicted == y_train_tensor).float().mean()
        train_accuracies.append(train_accuracy.item())

        best_model.eval()
        val_outputs = best_model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

        test_outputs = best_model(X_test_tensor)
        test_predicted = (test_outputs >= 0.5).float()
        test_accuracy = (test_predicted == y_test_tensor).float().mean()
        test_accuracies.append(test_accuracy.item())

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

