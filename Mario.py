# make sure to use python 3.8, use virtual enviornment and reopen vscode 
import gym.wrappers
import gym.wrappers.frame_stack
import gym.wrappers.gray_scale_observation
import gym.wrappers.resize_observation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros # install the VS code Developer kit?? refer to nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym # use pip install gym==0.23.1 , SMB does not work with new gym
import retro # use to import sonic
import DQN # our assignment 2 DQN modified for gym instead of gymnasium, may change later
from gym.spaces import Box

#create environment and agent + image preprocessing
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.reset()
# env = gym.wrappers.frame_stack.
env = gym.wrappers.gray_scale_observation(env)
env = gym.wrappers.resize_observation(env, shape= 84)

Buffer_Size = 5000
Mario = DQN.DQN(Buffer_Size, env)
print("Sucessfully Created agent:", Mario)

#HyperParameters:
max_episodes = 600
epsilon = .8
discount = 0.99
action = Mario.action


#Run DQN here
total_rewards = Mario.train(episodes=max_episodes, epsilon=epsilon , discount=discount,action_function=action, greedy=False )
#testing
total_rewards = Mario.train(episodes=1, epsilon=epsilon , discount=discount,action_function=action, greedy=True )

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close()
