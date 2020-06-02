#
#   Show the Uncertainty of a model
#

# Based on heteroskedastic data

# Importing the libraries

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# generate a batch of data
def gen_batch(batch_size = 32):
    
    # x in (-5, + 5)
    x = np.random.random(batch_size) * 10 - 5
    
    # linear function standard deviation of x
    sd = 0.05 + 0.1 * (x + 5)

    # y = mean + gaussian noise * sd
    y = np.cos(x) - 0.3 * x + np.random.randn(batch_size) * sd

    return x, y

# visualise the data
x, y = gen_batch(1024)
plt.scatter(x, y, alpha = 0.5)

# Model
class Model(nn.Module):
    def __init__(self):
    
        super(Model, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1))
    
        self.fc2 = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1))
  
    def forward(self, x):
        # return mean, log-variance
        return self.fc1(x), self.fc2(x)

model = Model()

# custom criterion
def criterion(outputs, targets):

    mu = outputs[0]
    v = torch.exp(outputs[1])

    # coefficient term
    c = torch.log(torch.sqrt(2 * np.pi * v))

    # exponent term
    f = 0.5 / v * (targets - mu) ** 2

    # mean log-likelihood
    nll = torch.mean(c + f)

    return nll

optimizer = torch.optim.Adam(model.parameters())

# training loop
n_epochs = 5000
batch_size = 128
losses = np.zeros(n_epochs)

for ep in range(n_epochs):

    x, y = gen_batch(batch_size)

    inputs = torch.from_numpy(x).float()
    targets = torch.from_numpy(y).float()

    # reshape
    inputs, targets = inputs.view(-1, 1), targets.view(-1, 1)

    # zero grad
    optimizer.zero_grad()

    # forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # store loss
    losses[ep] = loss.item()

    # print loss
    if ep % 1000 == 0:
        print(ep, losses[ep])
    
    # optimise
    loss.backward()
    optimizer.step()

# plot losses
plt.plot(losses)

# plot predictions
x, y = gen_batch(1024)
plt.scatter(x, y, alpha = 0.5)

inputs = torch.from_numpy(x).float()
targets = torch.from_numpy(y).float()

# reshape
inputs, targets = inputs.view(-1, 1), targets.view(-1, 1)

with torch.no_grad():
    outputs = model(inputs)
    yhat = outputs[0].numpy().flatten()
    sd = np.exp(outputs[1].numpy().flatten() / 2)

i = np.argsort(x)
plt.plot(x[i], yhat[i], linewidth = 3, color = 'red')
plt.fill_between(x[i], yhat[i] - sd[i], yhat[i] + sd[i], color = 'red', alpha = 0.5)
plt.show()