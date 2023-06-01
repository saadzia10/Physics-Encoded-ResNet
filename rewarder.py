import copy
import numpy as np

class Rewarder:

    def __init__(self):
        self.prev_action = None

    def calculate_reward(self, obs, obs_pre, action):
        sp = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        if self.prev_action is not None:
            reward = reward - (1 if np.abs(action['steer'] - self.prev_action['steer']) > 0.6 else 0)

        # Penalize if constant speed
        # if self.throttle:
        #     if obs["speedX"] == obs_pre["speedX"] == 0:
        #         reward -= 1

        self.prev_action =  copy.deepcopy(action)

        return reward
