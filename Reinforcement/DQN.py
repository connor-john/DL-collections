# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:18:45 2020

@author: Connor
@environment: pytorch
@references: 
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning
    cs.toronto.edu/~vmnih/docs/dqn.pdf
    
Naive DQN to solve cartpole game
"""

# Importing Libraries
import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Initialise
env = gym.make('CartPole-v0')

# Hyperparameters
gamma = 0.99
epsilon = 1.0
n_games = 1000
lr = 0.0001

# Batch History
scores = []
eps_history = []

# Linear Deep Q-Network
class LDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LDQN, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        actions = self.fc2(l1)
        
        return actions
        
# Agent class
class Agent():
    def __init__(self, input_dims, n_actions, lr = lr, gamma = gamma, epsilon = epsilon, eps_dec = 1e-5, eps_min = 0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        
        self.action_space = [i for i in range(self.n_actions)]
        
        # Impleting the Q value as a value of the Agent, not the entire DQN
        self.Q = LDQN(self.lr, self.n_actions, self.input_dims)
    
    # Using epsilon greedy action selection
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype = T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def decrease_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)
        
        q_pred = self.Q.forward(states)[actions]
        
        q_next = self.Q.forward(states_).max()
        
        q_target = reward + self.gamma * q_next
        
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrease_epsilon()
                
# Initialise Agent
agent = Agent(input_dims = env.observation_space.shape, n_actions = env.action_space.n)

# Plot learning curve
def plot_durations(x, scores, epsilons):
    fig = plt.figure()
    ax = fig.add_subplot(111, label = "1")
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    
    ax.plot(x, epsilons, color = "C0")
    ax.set_xlabel("Training steps", color = "C0")
    ax.set_ylabel("Epsilon", color = "C0")
    ax.tick_params(axis = 'x', colors = "C0")
    ax.tick_params(axis = 'y', colors = "C0")
    
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 100) : (t + 1)])
    
    ax2.scatter(x, running_avg, color = "C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color = "C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis = 'y', colors = "C1")
    
    plt.show()
       
def main():
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 50 == 0:
            mean_score = np.mean(scores[-100:])
            print('Episode {}\tAverage score (last 100 episodes): {:.2f}'.format(i, mean_score))
    
    x = [i+1 for i in range(n_games)]
    plot_durations(x, scores, eps_history)


main()
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        