import random
# Install required libraries
# Import required libraries
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
device = "cpu"
print(device)
CUDA_LAUNCH_BLOCKING = 1


class Net(nn.Module):
    def __init__(self, obs, action):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(obs, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.layer1 = nn.Linear(64 * 7 * 7, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN:
    # initialize values
    def __init__(self, N, env):
        # initialize environment
        self.env = env
        # initialize replay memory to capacity N
        self.replay = [None] * N
        self.capacity = N
        self.pointer = 0
        self.policy_net = Net(1, self.env.action_space.n).to(device)
        self.target_net = Net(1, self.env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync weights
        self.optimizer = optim.Adam(self.policy_net.parameters(), amsgrad=True)

        self.C = 50

    # Main training function
    def train(self, episodes, epsilon, gamma, action_function, greedy):
        total_reward = [0] * episodes
        for i in range(episodes):
            # initialize sequence S and preprocessed sequence o
            state, info = self.env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).transpose(0, 2)
            print(state.shape)
            done = False
            rewards = steps = 0
            # Decay epsilon after every episode
            eps = epsilon ** i if not greedy else 0
            self.env.reset()
            while not done:
                # Select action
                action_type = action_function(eps, state)

                # Execute action and observe reward
                observation, reward, terminated, truncated, info = self.env.step(action_type.item())
                next_state = torch.tensor(observation, device=device, dtype=torch.float32).transpose(0, 2)

                # Format next state
                done = terminated or truncated or steps > 500
                if done:
                    next_state = None
                # Add to total rewards for the episode
                rewards += reward
                # Encode action type for ease of use
                action_type = torch.tensor([action_type], device=device, dtype=torch.int64)
                # store transition in replay buffer
                transition = state, action_type, next_state, reward
                state = next_state

                self.replay[self.pointer % self.capacity] = transition
                self.pointer += 1

                # When terminated store the last value found
                if done:
                    transition = state, action_type, None, reward
                    self.replay[self.pointer % self.capacity] = transition
                    self.pointer += 1

                batch_size = 128
                # Run the replay function if there is enough transitions
                if batch_size < self.pointer:
                    self.replay_function(gamma ** steps, batch_size)

                # Every C steps batch
                if steps % self.C == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                steps += 1
            print("Episode: ", i, " Reward: ", rewards)
            total_reward[i] = rewards
        return total_reward

    # Determine the action for the warehouse environment
    def action(self, epsilon, state):
        if np.random.rand() < epsilon:
            action_type = self.env.action_space.sample()
        else:
            # select max(Q)
            with torch.no_grad():
                action_type = self.policy_net(state.unsqueeze(1)).max(1).indices.item()
        return torch.tensor([[action_type]], device=device, dtype=torch.long)

    def replay_function(self, gamma, batch_size):
        if self.pointer < self.capacity:
            temp = self.replay[:self.pointer]
            sample = random.sample(temp, k=batch_size)
        else:
            sample = random.sample(self.replay, k=batch_size)
        Q_list = torch.tensor([], device=device)
        target_val = torch.tensor([], device=device)
        action_list = torch.tensor([], dtype=torch.int64, device=device)
        for state, action, next_state, reward in sample:
            if next_state is None:
                Q_list = torch.cat((Q_list, self.policy_net(state)))
                # Make an actions array
                action_list = torch.cat((action_list, action), 0)
                # Calculate updated Q value
                Q_val = torch.tensor([reward], device=device)
                # Add value to expected target list
                target_val = torch.cat((target_val, Q_val))

            else:
                # Take entire Q row
                Q_list = torch.cat((Q_list, self.policy_net(state)))
                # Make an actions array
                action_list = torch.cat((action_list, action), 0)
                # Take max expected Q from the target network
                max_expected = self.target_net(next_state).max(1).values
                print(max_expected)
                # Calculate updated Q value
                Q_val = torch.tensor([(max_expected * gamma) + reward], device=device)
                # Add value to expected target list
                target_val = torch.cat((target_val, Q_val))

        # Apply the action list to get real expected Q values
        selected_q_values = Q_list.gather(1, action_list.unsqueeze(1))

        loss_function = nn.MSELoss()
        loss = loss_function(selected_q_values, target_val.unsqueeze(1))
        # backprop
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    # Save the current weights
    def save(self, filename):
        with open("pickles/" + filename, 'wb') as file:
            pickle.dump(self.policy_net, file)
