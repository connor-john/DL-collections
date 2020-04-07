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

# RNN model
class RNN(nn.Module):
    def __init__(self, vocab_dim, seq_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, vocab_dim)
    
    def forward(self, x, prev_state):
        e = self.embedding(x)
        output, state = self.lstm(e, prev_state)
        out = self.fc1(output)
        
        return out, state
    
    # reset hidden state and memory state
    def reset_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))

# Hyper parameters
lr = 0.001

# Initialise
rnn = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = lr)