import os
import sys
import gym
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

NUMBER_OF_EXAMPLES = 1
HORIZON = 300
ENV_NAME = "Acrobot-v1"


class FuzzyController:
    def __init__(self):
        self.clear()

    def clear(self):
        """
        initialize and set fuzzy term and rules
        """
        self.position = ctrl.Antecedent(np.arange(-1.2, 0.6, 0.01), "position")
        self.velocity = ctrl.Antecedent(np.arange(-0.7, 0.07, 0.01), "velocity")
        self.power_coefficient = ctrl.Consequent(np.arange(-1, 1, 0.01), "power coefficient")

        self.position["low"] = fuzz.trimf(self.position.universe, [-1.2, -1.2, -0.3])
        self.position["mid"] = fuzz.trimf(self.position.universe, [-0.75, -0.3, 0.15])
        self.position["high"] = fuzz.trimf(self.position.universe, [0, 0.6, 0.6])

        self.velocity["low"] = fuzz.trimf(self.velocity.universe, [-0.7, -0.7, -0.35])
        self.velocity["mid"] = fuzz.trimf(self.velocity.universe, [-0.7, -0.35, 0])
        self.velocity["high"] = fuzz.trimf(self.velocity.universe, [0, 0.07, 0.07])

        self.power_coefficient["low"] = fuzz.trimf(self.power_coefficient.universe, [-1, -1, 0])
        self.power_coefficient["mid"] = fuzz.trimf(self.power_coefficient.universe, [-0.5, 0, 0.5])
        self.power_coefficient["high"] = fuzz.trimf(self.power_coefficient.universe, [0, 1, 1])

        rules = [ctrl.Rule(self.velocity["low"], self.power_coefficient["low"]),
                 ctrl.Rule(self.velocity["mid"], self.power_coefficient["mid"]),
                 ctrl.Rule(self.velocity["high"] & self.position["low"], self.power_coefficient["high"]),
                 ctrl.Rule(self.velocity["high"] & self.position["mid"], self.power_coefficient["high"]),
                 ctrl.Rule(self.velocity["high"] & self.position["high"], self.power_coefficient["low"]),
                 ]

        self.force_ctrl = ctrl.ControlSystem(rules)
        self.ctrl_sim = ctrl.ControlSystemSimulation(self.force_ctrl)

        self.output = 0.0

    def update(self, feedback_value):
        """
        update self.power_coefficient
        """
        self.ctrl_sim.input["position"] = feedback_value[0]
        self.ctrl_sim.input["velocity"] = feedback_value[1]
        self.ctrl_sim.compute()

        print("after fuzzy : {} ".format(self.ctrl_sim.output["power coefficient"]))
        self.output = self.ctrl_sim.output["power coefficient"]


def plot_results(rewards_array):
    fig, axs = plt.subplots()
    axs.plot(rewards_array)
    axs.set_title('rewards')
    plt.show()


if __name__ == '__main__':
    lib_path = os.path.abspath(os.path.join(sys.path[0], ".."))
    sys.path.append(lib_path)
    Ctl = FuzzyController()
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
    plot_results(np.array(rewards))

