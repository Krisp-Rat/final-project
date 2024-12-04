# taken from assignment 2 
# CHANGES MADE FROM GYMNASIUM TO GYM
# removed info var as reset does not return info
# removed truncated as truncated does not exist: only done

# CHANGES THAT NEED TO BE DONE
# Input is now a image, your going to have to modify it to accept the new state in both the transition array and the neural netork

# Install required libraries
# Import required libraries
import random
import gym
from gym import spaces
# import matplotlib.pyplot as plt
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
        self.pointer = 0
        self.policy_net = Net(1, self.env.action_space.n).to(device)
        self.target_net = Net(1, self.env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync weights

        self.optimizer = optim.AdamW(self.policy_net.parameters(), amsgrad=True)  # auto learning rate
        # big replay memory:
        self.size = N
        shape = (N, 1, self.env.observation_space.shape[0], self.env.observation_space.shape[1])
        self.state_mem = torch.zeros(shape, dtype=torch.float32, device=device)
        self.next_state_mem = torch.zeros(shape, dtype=torch.float32,
                                          device=device)
        self.action_mem = torch.zeros(self.size, dtype=torch.int64, device=device)
        self.reward_mem = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.done_mem = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.pointer = 0

    def append(self, state, action, reward, next_state, done):
        i = self.pointer % self.size  # get index
        self.state_mem[i] = state
        self.next_state_mem[i] = next_state
        self.reward_mem[i] = reward
        self.done_mem[i] = 1 - int(done)
        self.action_mem[i] = action
        self.pointer += 1

    def sample(self, batch):
        mem = min(self.pointer, self.size)  # get range to choose mem from
        batch = np.random.choice(mem, batch)  # choose random indices
        states = self.state_mem[batch]
        next_states = self.next_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        done = self.done_mem[batch]
        return states, actions, rewards, next_states, done

    # Main training function
    def train(self, episodes, epsilon, discount, action_function, greedy):
        total_reward = [0] * episodes
        for i in range(episodes):
            if i % 5 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            # initialize sequence S and preprocessed sequence o
            state, info = self.env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).transpose(0, 2)
            done = False
            rewards = step = 0
            eps = epsilon ** i if not greedy else 0
            while not done:
                # Select action
                action_type = action_function(state, eps)
                observation, reward, terminated, truncated, info = self.env.step(action_type.item())
                next_state = torch.tensor(observation, device=device, dtype=torch.float32).transpose(0, 2)
                done = terminated or truncated

                # store transition in replay buffer
                self.append(state=state, action=action_type, reward=reward, next_state=next_state, done=done)

                # if the episode is over set the next state to none
                done = terminated or truncated
                if done:
                    next_state = None

                rewards += reward
                state = next_state
                if self.pointer%10 == 0 :
                    self.r(discount)
                step += 1

                if greedy:
                    self.env.render()
                if step > 500:
                    done = True

            # Decay epsilon after every episode
            # epsilon *= epsilon
            total_reward[i] = rewards
        return total_reward
        # Determine the action for the warehouse environment

    def action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action_type = self.env.action_space.sample()
        else:
            # select max(Q)
            with torch.no_grad():
                action_type = self.policy_net(state.unsqueeze(1)).max(1).indices.item()
        return torch.tensor([[action_type]], device=device, dtype=torch.long)

    def r(self, discount):
        BATCH_SIZE = 256
        if self.pointer < BATCH_SIZE:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, dones = self.sample(BATCH_SIZE)

        # Convert to tensors
        state_batch = torch.tensor(states, dtype=torch.float32, device=device)
        action_batch = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32, device=device)
        done_mask = torch.tensor(dones, dtype=torch.float32, device=device)  # zeros out terminated

        # Calculate state-action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calculate next state values
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[done_mask.bool()] = self.target_net(next_state_batch[done_mask.bool()]).max(1).values

        # Expected Q values
        expected_state_action_values = (next_state_values * discount) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Save/load the current weights

    def load(self, filename):
        with open(filename, 'rb') as file:
            # Load the state dict from the file
            state_dict = pickle.load(file)
            # Apply the state dict to the model
            self.policy_net.load_state_dict(state_dict)
            # Optionally, set the model to evaluation mode
            self.policy_net.eval()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.policy_net.state_dict(), file, protocol=pickle.HIGHEST_PROTOCOL)
