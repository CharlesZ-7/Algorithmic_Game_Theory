
from typing import Any
import gymnasium as gym
from rl import DQNAgent




class MyEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()

        self.observation_space = gym.spaces.Box(

        )
        self.action_space = gym.spaces.Discrete(1)
        self.model = DQNAgent(self)

    def step(self, action):
        # making some dummy stuff happen...
        pass

    def reset(self):
        pass









if __name__ == "__main__":

    print("hello world!")


    env = MyEnv()
    env.reset()
    env.step()






# import random, time, gym, cv2
# from PIL import ImageGrab
# import pyautogui

# class MyEnv(gym.Env):
#     def __init__(self):
#         super().__init__(self)

#         self.observation_space = gym.Box()
#         self.action_space = gym.Discrete(N_ACTIONS) #number of controls
        
#         self.model = DQNAgent()
#         self.previous_observation = None

#     def step(self, action):
#         # Conditional logic for what to do with actions
#         # an example
#         if action == 0:
#             pyautogui.press('w') # Go forwards
#             reward = 1
        
#         shot = np.array(ImageGrab.grab(bbox=("""x, y, width, height of game window""")))
#         gray = cv2.cvtColor(Screen, cv2.COLOR_BGR2GRAY)
#         if self.previous_observation is not None:
#             self.model.update_replay_memory((self.previous_observation, action, reward, gray))

#         self.prevoius_observation = gray

#         # check if the player has lost, and call self.reset()

#         return observation, action, reward, {}

#     def reset(self):
#         # reset the game (re-open it, or something like that)

# env = MyEnv()
# epsilon = 0.1
# decay = 0.99998
# min = 0.001
# steps = 60000

# # open the game here
# # ...

# for i in range(0, steps):
#     if random.random() < epsilon:
#         env.step(env.action_space.sample())
#         elipson *= decay
#     else:
#         try:
#             env.step(env.model.get_qs(env.previous_observation))

# env.reset()
# env.model.save('models/player.h5')
# # close the game here
# # ...