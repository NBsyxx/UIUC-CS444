# ys2674
import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.criterion = nn.HuberLoss(reduction='mean')
        
        # Initialize a target network and initialize the target network to the policy net
        ### CODE ###
        self.target_net = DQN(action_size)
        self.target_net.to(device)
        self.update_target_net()


    def load_policy_net(self, path):
        self.policy_net.load_state_dict(torch.load(path))          

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        ### CODE ###
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action if less than epsilon 
            action = np.random.choice(self.action_size)
        else:
            ### CODE ####
            # Choose the best action
            with torch.no_grad():
#                 print(state.shape)
                current_state = torch.from_numpy(state).unsqueeze_(dim=0)
                # needs to be on device
                current_state = current_state.to(device)
#                 print(current_state.shape)
                Q_values = self.policy_net(current_state)
                action = torch.argmax(Q_values).item()
#                 print(Q_values)
#                 print(action)
        return action

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)
        
        # Your agent.py code here with double DQN modifications
        ### CODE ###
        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        Q_values_all_actions_current = self.policy_net(states)
        # gathers all values according to action
        Q_values_current = Q_values_all_actions_current.gather(1, actions.unsqueeze(dim=1))
#         print("actions",actions.unsqueeze(dim=1))
#         print("combQ",current_Q_values)

        # Compute Q function of next state
        ### CODE ####
        with torch.no_grad():
            # print(type(next_states))
            next_states = torch.from_numpy(next_states).to(device)
            Q_values_all_actions_next = self.policy_net(next_states)

            # Find maximum Q-value of action at next state from policy net
            ### CODE ####
            Q_values_maximum_next = Q_values_all_actions_next.max(dim=1)[0]
            
        # expected Q values for next state: lambda * mask * Q_values_maximum_next
        E_Q_values = self.discount_factor * mask.to(device) * Q_values_maximum_next + rewards
            
        # Compute the Huber Loss
        ### CODE ####
        # init self.criterion while boot
        loss = self.criterion(Q_values_current.squeeze(dim=-1), E_Q_values)

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
     
        