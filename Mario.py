# make sure to use python 3.8, use virtual enviornment (to make terminal also 3.8) and reopen vscode 
import gym.wrappers
import gym.wrappers.frame_stack
from gym.wrappers import GrayScaleObservation 
from gym.wrappers import ResizeObservation # these things also use cv2, pip install it
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros # install the VS code Developer kit?? refer to nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym # use pip install gym==0.23.1 , gym_super_mario_bros does not work with new gym
import retro # use to import sonic, #don't worry about it for now
import DQN # our assignment 2 DQN modified for gym instead of gymnasium, may change later
from gym.spaces import Box


# CHANGES THAT NEED TO BE DONE:
    # https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html for reference, do not copy. 
    # we need to processes the image to match what is shown here, you can also use your own size if you find a better one

#create environment and agent + image preprocessing
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state = env.reset()
print("inital state size", state.shape )

# IMAGE PROCESSING HERE
# env = gym.wrappers.frame_stack.
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape= 84)
state = env.reset()
print("mod state size", state.shape )


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
#testing network
total_rewards = Mario.train(episodes=1, epsilon=epsilon , discount=discount,action_function=action, greedy=True )


# testing if simulation works for MARIO, this should work fine, we just need to fix the NERUAL NETWORK for images
done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close()
