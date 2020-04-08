# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:03:14 2020

@author: Connor
Implementation of RNN in PyTorch
@references: 
    https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""

# Importing the Libraries
import numpy as np
import torch
import torch.nn as nn

# Hyper parameters
lr = 0.001
n_epochs = 1000
batch_size = 100
text_len = 200 
all_characters = string.printable
n_characters = len(all_characters)

# RNN model
class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers = 1):
        super(RNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, hidden_layers)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    # returns output and hidden state
    def forward(self, x, hidden):
        #batch_size = x.size(0)
        e = self.embedding(x)
        output, hidden = self.lstm(e, hidden)
        output = self.fc1(output)
        
        return output, hidden
    
    # reset hidden state and memory state
    def reset_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))

# Read file
def read_file(filename):
    file = open(filename, 'r').read() # should be simple plain text file
    return file, len(file)

# Convert string to tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Initialise
rnn = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = lr)
filename = 'input.txt'

# train model    
def train(x, target):
    hidden = rnn.reset_state(batch_size)
    hidden = hidden.cuda()
    
    rnn.zero_grad()
    loss = 0
    
    for c in range(text_len):
        output, hidden = rnn(x, hidden)
        loss =+ criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    return loss.data[0] / text_len

# Main
def main():
    