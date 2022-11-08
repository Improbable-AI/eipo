import gym
import numpy as np
from PIL import Image
import cv2

class Grayscale(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        super(Grayscale, self).__init__(env)
        width, height, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0., high=255., shape=(width, height, 1), dtype=np.uint8)

    def observation(self, obs):
        img = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        return np.expand_dims(img, axis=2) # (w, h, c)

class ResizeFull(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        super(ResizeFull, self).__init__(env)
        self._shape = (84, 84)
        self.observation_space = gym.spaces.Box(low=0., high=255., shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):
        img = cv2.resize(obs, self._shape, interpolation=cv2.INTER_NEAREST)
        return img