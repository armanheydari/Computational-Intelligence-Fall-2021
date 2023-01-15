import os
import sys
import gym
import matplotlib.pyplot as plt
from PID import *
import numpy as np

NUMBER_OF_EXAMPLES = 1
HORIZON = 300
ENV_NAME = "MountainCarContinuous-v0"


def plot_results(rewards_array, distance_from_goal_array):
    fig, axs = plt.subplots(2)
    axs[0].plot(rewards_array)
    axs[0].set_title('rewards')
    axs[1].plot(distance_from_goal_array)
    axs[1].set_title('distance from goal')
    plt.show()


lib_path = os.path.abspath(os.path.join(sys.path[0], ".."))
sys.path.append(lib_path)
Ctl = PID()
graph = []
env = gym.make(ENV_NAME)
distance_from_goal = []
rewards = []
for i_episode in range(NUMBER_OF_EXAMPLES):
    observation = env.reset()
    for t in range(HORIZON):
        env.render()
        print(observation)
        feedback = env.state
        graph.append(feedback[0])
        print("f is : ", feedback)
        Ctl.update(feedback)
        action = [Ctl.output]
        print("action ", action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        distance_from_goal.append(0.45 - observation[0])
env.close()
plot_results(np.array(rewards), np.array(distance_from_goal))
