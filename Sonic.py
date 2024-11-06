# make sure to use python 3.8, use virtual enviornment and reopen vscode 
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros # install the VS code Developer kit?? refer to nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym # use pip install gym==0.23.1 , SMB does not work with new gym
import retro # use to import sonic
import DQN # our assignment 2 DQN modified for gym instead of gymnasium, may change later
