# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:24:44 2020

@author: Connor
@environment: Using a pytorch GPU environment

Movie recommender using a Stacked Auto Encoder 
"""

# Stacked Auto Encoder

# Importing the Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset (using MovieLens GroupLens dataset)
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the datasets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array for torch function
def convert(data):
    new_data = []
    for user in range(1, nb_users + 1):
        movies = data[:, 1][data[:, 0] == user]
        ratings = data[:, 2][data[:, 0] == user]
        f_ratings = np.zeros(nb_movies)
        f_ratings[movies - 1] = ratings
        new_data.append(list(f_ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting data into Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the NN
class SAE(nn.module):
    def __init__(self, ):
        

        
        
        
        
        
        
        