from collections import deque
import gym
from gym import spaces
import numpy as np

class FrameSkip(gym.Wrapper):
    """
    Repeat a single action through four n frames.
    """
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info

class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to np.ndarray before being passed to the model.
    :param frames: ([int] or [float]) environment frames
    """
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class FrameStack(gym.Wrapper):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping. 

    n is the length of the buffer, and number of observations stacked.
    skip is the number of steps between buffered observations (min=1).

    n.b. first obs is the oldest, last obs is the newest.
         the buffer is zeroed out on reset.
         *must* call reset() for init!
    """
    def __init__(self, env=None, n=4, skip=4, shape=(84, 84),
                    channel_last=False, maxFrames=True):
        super(BufferedObsEnv, self).__init__(env)
        self.obs_shape = shape
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.maxFrames = maxFrames
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
    
        self.ch_axis = -1 if channel_last else 0 # should be 0 if (c, h, w) after applying PyTorchImage wrapper
        shape = shape + (n,) if channel_last else (n,) + shape
        self.observation_space = spaces.Box(0.0, 255.0, shape)
        self.observation_space.high[...] = 1.0
        self.scale = 1.0 / 255

    def observation(self, obs):
        obs = self._max_recent(obs) # take max between last two frames
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs = self._max_recent(self.env.reset())
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _max_recent(self, obs):
        self.obs_buffer.append(obs)
        if self.maxFrames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        return max_frame

class PytorchImage(gym.ObservationWrapper):
    """
    Switch image observation from (h, w, c) to (c, h, w) 
    which is required for the CNNs and other Pytorch things.
    """
    def __init__(self, env):
        super(PytorchImage, self).__init__(env)
        current_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(current_shape[-1], current_shape[0], current_shape[1]))

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))

class NoExtrinsicReward(gym.RewardWrapper):
    """
    Remove external reward for experiments where you want
    to use intrinsic reward only.
    """
    def __init__(self, env):
        super(NoExtrinsicReward, self).__init__(env)

    def reward(self, reward):
        return 0.0

class NoNegativeReward(gym.RewardWrapper):
    """
    Remove negative rewards and zero them out. This can apply
    to living penalties for example.
    """
    def __init__(self, env):
        super(NoNegativeReward, self).__init__(env)

    def reward(self, reward):
        if reward < 0:
            return 0
        else:
            return reward


