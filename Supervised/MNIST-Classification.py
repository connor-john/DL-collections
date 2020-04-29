# Importing the Libraries
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper params
hidden_dim = 128
batch_size = 128
n_epochs = 10

# Get Data -- Using MNIST dataset
train_data = torchvision.datasets.MNIST(
    root = '.', 
    train = True, 
    transform = transforms.ToTensor(),
    download = True)

test_data = torchvision.datasets.MNIST(
    root = '.', 
    train = False, 
    transform = transforms.ToTensor(),
    download = True)

# Model
model = nn.Sequential(
    nn.Linear(784, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 10)
)

# Initialise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Prep data
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

# Training
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for i in range(n_epochs):
  
  train_loss = []
  
  for inputs, targets in train_loader:

    inputs, targets = inputs.to(device), targets.to(device)
    inputs = inputs.view(-1, 784)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
      
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

  train_loss = np.mean(train_loss)
  
  test_loss = []

  for inputs, targets in test_loader:

    inputs, targets = inputs.to(device), targets.to(device)
    inputs = inputs.view(-1, 784)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    test_loss.append(loss.item())

  test_loss = np.mean(test_loss)

  train_losses[i] = train_loss
  test_losses[i] = test_loss
    
  print(f'epoch {i+1} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}')

# test accuracy
n_correct = 0.
n_total = 0.

for inputs, targets in test_loader:

  inputs, targets = inputs.to(device), targets.to(device)
  inputs = inputs.view(-1, 784)

  outputs = model(inputs)

  _, pred = torch.max(outputs, 1)

  n_correct += (pred == targets).sum().item()
  n_total += targets.shape[0]

test_acc = n_correct / n_total
print(f'test_accuracy: {test_acc:.4f}')
                                           
   