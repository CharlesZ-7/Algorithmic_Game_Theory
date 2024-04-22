


from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np

class MyEnv(gym.Env):



    def __init__(self, env_config):

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(5)

    def reset(self, seed=None, options=None):

        # obs = gym.spaces.Discrete(1)
        # obs.

        # <obs>, <info: dict>
        return np.int64(1), {}

    def step(self, action):

        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
        return np.int64(1), 0.5, True, False, {}
    
    
    
