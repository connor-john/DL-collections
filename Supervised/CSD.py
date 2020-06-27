# Cauchy-Schwarz Divergence
# Custom loss for CNN

# Importing the libraries
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper params
batch_size = 32
lr = 0.0005
num_epochs = 128

# Getting the Data
# Using MNIST
data_train = torchvision.datasets.MNIST(
    './data/mnist',
    train = True,
    download = True,
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(2),
        torchvision.transforms.ToTensor()
    ])
)

data_test = torchvision.datasets.MNIST(
    './data/mnist',
    train = False,
    download = True,
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(2),
        torchvision.transforms.ToTensor()
    ])
)

# Data loaders
train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4
)

test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4
)

# Model
class CNN(torch.nn.Module):
    
    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Sequential(
            # layer 1
            torch.nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), stride=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride=2),
            # layer 2
            torch.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride=2),
            # layer 3
            torch.nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5, 5), stride=1, bias=True),
            #torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size = (2, 2), stride=2)
        )
        
        self.fc1 = torch.nn.Sequential(
            # layer 1
            torch.nn.Linear(
                in_features = 120,
                out_features = 84,
                bias=True
            ),
            torch.nn.ReLU(),
            # classifier
            torch.nn.Linear(
                in_features = 84,
                out_features= 10,
                bias=True
            ),
            torch.nn.Softmax(dim = 1)
        )

    def forward(self, input):
        x = self.conv1(input)
        # flatten
        x = x.view(input.size(0), -1)
        x = self.fc1(x)
        return x

# Cauchy-Schwarz Divergence

def one_hot_encode(input):
  a = input.cpu().numpy()
  # 10 for output dim 0-9
  b = np.zeros((a.size, 10))
  # set 1 where each element in a
  b[np.arange(a.size), a] = 1

  return torch.from_numpy(b).float().to(device)

class CSD(torch.nn.Module):

  def __init__(self):
    super(CSD, self).__init__()

  def forward(self, outputs, target):
    y = one_hot_encode(target)
    nominator = torch.sum(torch.mm(outputs, y.t()), dim = 1)
    denominator = torch.norm(outputs, 2) * torch.norm(y, 2)

    return torch.mean(-1 * torch.log(nominator / denominator))

# initialise
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = CSD()

# training
steps = len(train_loader) // batch_size
model.train(True)

for e in range(num_epochs):
  
  total_loss = 0
  s = 0

  for i, (images, labels) in enumerate(train_loader):

    if i == steps:
      break
    
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    total_loss += loss

    s += 1
  
  assert s == steps, "steps: {} != {}".format(steps, s)

  out_loss = total_loss / steps

  print(f'epoch: {e}/{num_epochs} | loss: {out_loss:.4f}')


# Evaluation
avg_loss = 0
avg_acc = 0

model.train(False)

with torch.no_grad():
    
    steps = 0

    for images, labels in test_loader:
        
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        avg_loss += criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        avg_acc += preds.eq(labels).sum().item()
        
        steps += 1

out_loss = avg_loss / steps
out_acc = avg_acc / (steps * batch_size)
print(f'loss: {out_loss:.2f} | acc: {out_acc:.2%} ')