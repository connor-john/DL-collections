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
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()
    
    def reset(self):
        self.action_episode = torch.Tensor([]) 
        self.reward_episode = []
        

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.5),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

# Initialise
policy = PolicyModel()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Select action given state
def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(state)
    distribution = Categorical(state)
    action = distribution.sample()
    
    # Add probability of action to history
    policy.action_episode = torch.cat([
        policy.action_episode,
        distribution.log_prob(action).reshape(1)
    ])
    
    return action

# Update policy using Monte-Carlo   
def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.action_episode, rewards).mul(-1), -1))

    # Update Policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save episode history
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.reset()

    
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
    for ep in range(num_episodes):
        state = env.reset()
        for t in range(1000): # do not run infinitely during training
            action = select_action(state)
            
            #env.render()
            
            state, reward, done, _ = env.step(action.item())
            
            # Save reward
            policy.reward_episode.append(reward)
            
            if done:
                #plot_durations(optimizer, policy)
                break
        
        # For batch learning approach
        # if ep > 0 and ep % batch_size == 0:
        #   update_policy()
        
        update_policy()
        
        # Update scores
        episode_durations.append(t)
        mean_score = np.mean(episode_durations[-100:])
        
        # Outputs
        if ep % 50 == 0:
            print('Episode {}\tAverage length (last 100 episodes): {:.2f}'.format(ep, mean_score))
        
        if mean_score > env.spec.reward_threshold:
            print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps.".format(ep, mean_score, t))
            break

main()