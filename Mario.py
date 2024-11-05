# make sure to use python 3.8, use virtual enviornment and reopen vscode 
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros # install the VS code Developer kit?? refer to nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym # use pip install gym==0.23.1
import retro
import pickle
import torch


#create environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


#put DQN here

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
