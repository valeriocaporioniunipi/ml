from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import monk_data, abs_path

# Load MONK dataset
features_test, targets_test = monk_data(abs_path('monks-3.test', 'data'))
features, targets = monk_data(abs_path('monks-3.train', 'data'))

# Standardize the features
scaler = OneHotEncoder(sparse_output=False)
X_train = scaler.fit_transform(features)
X_test = scaler.transform(features_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, targets, test_size=0.2, random_state=42
)

# Convert labels to float (ensure they are 0 or 1)
y_train = targets.astype(float).values.reshape(-1, 1)
y_test = targets_test.astype(float).values.reshape(-1, 1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
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
        nn.init.zeros_(m.bias)

# Grid search parameters
etas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
lambdas = [0.005, 0.01, 0.1]
alphas = [0.001, 0.01, 0.1, 0.5, 1.0]
best_model = None
best_accuracy = 0

for eta, lambda_, alpha in product(etas, lambdas, alphas):
    model = BinaryClassificationNN(X_train.shape[1], 16)
    model.apply(weights_init)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=lambda_)
    
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
        test_outputs = model(X_test_tensor)
        test_predicted = (test_outputs >= 0.5).float()
        test_accuracy = (test_predicted == y_test_tensor).float().mean().item()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_params = {'eta': eta, 'lambda': lambda_, 'alpha': alpha}

print(f'Best Params: {best_params}, Best Accuracy: {best_accuracy:.4f}')

# Final training with best params
optimizer = optim.Adam(best_model.parameters(), lr=best_params['eta'], weight_decay=best_params['lambda'])
criterion = nn.BCELoss()
train_losses = []
val_losses = []
train_accuracies = []
test_accuracies = []
best_model.apply(weights_init)

for epoch in range(epochs):
    best_model.train()
    optimizer.zero_grad()
    train_outputs = best_model(X_train_tensor)
    train_loss = criterion(train_outputs, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    train_losses.append(train_loss.item())
    with torch.no_grad():
        train_predicted = (train_outputs >= 0.5).float()
        train_accuracy = (train_predicted == y_train_tensor).float().mean()
        train_accuracies.append(train_accuracy.item())

        best_model.eval()
        test_outputs = best_model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        val_losses.append(test_loss.item())
        test_predicted = (test_outputs >= 0.5).float()
        test_accuracy = (test_predicted == y_test_tensor).float().mean()
        test_accuracies.append(test_accuracy.item())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Test Loss', linestyle='--')
plt.title('Train vs Test Loss')
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

