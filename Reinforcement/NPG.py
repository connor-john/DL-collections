# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:02:14 2020

@author: Connor
@environment: pytorch
@references: 
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning
    https://karpathy.github.io/2016/05/31/rl/
    
Monte-Carlo Policy Gradient approach to solve cartpole problem

"""

#importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import numpy as np
import gym

# Setting up env
episode_durations = []
env = gym.make('CartPole-v1')

# Hyperparameters
num_episodes = 1000
gamma = 0.99
learning_rate = 0.01
batch_size = 5

# Policy Model
class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.state_pool = env.observation_space.shape[0]
        self.action_pool = env.action_space.n
        
        self.l1 = nn.Linear(self.state_pool, 128, bias = False)
        self.l2 = nn.Linear(128, self.action_pool, bias = False)
        
        self.gamma = gamma
        
        # Batch History
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        self.reward_pool = []
        self.loss_pool = []
    
    def forward(self, x):
        model = torch.nn.Sequential(
                self.l1,
                nn.Dropout(p = 0.6), # Adding dropout for overfitting
                nn.ReLU(),
                self.l2,
                nn.Softmax(dim = 1)
                )
        return model(x)

# Initialise
policy = PolicyModel()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Select action given state
def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    cat = Categorical(state)
    action = cat.sample()
    
    # Add probability of action to history
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, cat.log_prob(action)])
    else:
        policy.policy_history = (cat.log_prob(action))
    return action

# Update policy using Monte-Carlo   
def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)
    
    # Scale rewards     
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    
    # Update Policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update History
    policy.loss_pool.append(loss.data[0])
    policy.reward_pool.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []
    
# Plot duration curve (taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
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

def main():
    running_reward = 10
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        for t in range(1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action.data[0])
            
            # Save reward
            policy.reward_episode.append(reward)
            
            if done:
                episode_durations.append(t + 1)
                #plot_durations(optimizer, policy)
                break
        
        running_reward = (running_reward * 0.99) + (t * 0.01)
        
        if ep > 0 and ep % batch_size == 0:
            update_policy()
        
        # Outputs
        if ep % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(ep, t, running_reward))
        
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
            break

main()