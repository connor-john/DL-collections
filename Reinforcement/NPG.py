# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:02:14 2020

@author: Connor
@environment: pytorch
Naive Policy Gradient approach to solve cartpole problem
"""

#importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym

# Policy Model
class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc2 = nn.Linear(36, 1)
    
    def forward(self, x):
        x = Fun.relu(self.fc1(x))
        x = Fun.relu(self.fc2(x))
        x = Fun.sigmoid(self.fc3(x))
        return x

def play_one(env, policy_m, state, gamma):
    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    
    for t in count():
        prob = policy_m(state)
        m = Bernoulli(prob)
        action = m.sample()
    
        action = action.data.numpy().astype(int)[0]
        next_state, reward, done, _ = env.step(action)
        env.render(mode='rgb_array')
    
        if done:
            reward = 0
    
        state_pool.append(state)
        action_pool.append(float(action))
        reward_pool.append(reward)
    
        state = next_state
        state = torch.from_numpy(state).float()
        state = Variable(state)
    
        steps += 1
    
        if done:
            episode_durations.append(t + 1)
            plot_durations()
        
        
def update_policy(steps, gamma, optimizer, state_pool, action_pool, reward_pool, policy_m):
    running_add = 0
    for i in reversed(range(steps)):
        if reward_pool[i] == 0:
            running_add = 0
        else:
            running_add = running_add * gamma + reward_pool[i]
            reward_pool[i] = running_add
    
    reward_mean = np.mean(reward_pool)
    reward_std = np.std(reward_pool)
    for i in range(steps):
        reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
    
    optimizer.zero_grad()
    
    for i in range(steps):
        state = state_pool[i]
        action = Variable(torch.FloatTensor([action_pool[i]]))
        reward = reward_pool[i]
        probs = policy_m(state)
        m = Bernoulli(probs)
        loss = -m.log_prob(action) * reward  
        loss.backward()
    

def main():
    # Plot duration curve (taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
    episode_durations = []
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    # Initialise
    num_episodes = 5000
    batch_size = 5
    learning_rate = 0.01
    env = gym.make('CartPole-v0')
    policy_m = PolicyModel()
    optimizer = torch.optim.RMSprop(policy_m.parameters(), lr=learning_rate)
    
    
    for ep in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state).float
        state = Variable(state)
        env.render(mode='rgb_array')
        play_one()
        if ep > 0 and ep % batch_size == 0:
            update_policy()
        

if __name__ == '__main__':
    main()