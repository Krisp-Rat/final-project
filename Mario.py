# make sure to use python 3.8, use virtual enviornment (to make terminal also 3.8) and reopen vscode 
from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation  # these things also use cv2, pip install it
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros  # install the VS code Developer kit?? refer to nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# import retro # use to import sonic, #don't worry about it for now
import DQN  # our assignment 2 DQN modified for gym instead of gymnasium, may change later
import torch

# CHANGES THAT NEED TO BE DONE:
# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html for reference, do not copy.
# we need to process the image to match what is shown here, you can also use your own size if you find a better one

# create environment and agent + image preprocessing
env = gym_super_mario_bros.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode="human",
                                max_episode_steps=300)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# IMAGE PROCESSING HERE
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=(64, 64))
state, info = env.reset()
env.render()

Buffer_Size = 5000
Mario = DQN.DQN(Buffer_Size, env)
print("Successfully Created agent")

# HyperParameters:
max_episodes = 1
epsilon = .8
discount = 0.99
action = Mario.action

# Run DQN here
total_rewards = Mario.train(episodes=max_episodes, epsilon=epsilon, discount=discount, action_function=action,
                            greedy=False)
print("Completed One episode of training")
# testing network
total_greedy_rewards = Mario.train(episodes=1, epsilon=epsilon, discount=discount, action_function=action, greedy=True)
print("Completed one greedy episode")

# testing if simulation works for MARIO, this should work fine, we just need to fix the NERUAL NETWORK for images
# done = False
# for step in range(5000):
#     if done:
#         state, info = env.reset()
#     state, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     done = terminated or truncated
#     env.render()
# env.close()

# @misc{gym-super-mario-bros,
#   author = {Christian Kauten},
#   howpublished = {GitHub},
#   title = {{S}uper {M}ario {B}ros for {O}pen{AI} {G}ym},
#   URL = {https://github.com/Kautenja/gym-super-mario-bros},
#   year = {2018},
# }
