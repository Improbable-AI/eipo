
from mpi4py import MPI
import multiprocessing
import gym
import numpy as np
import cv2
from copy import deepcopy

from rlpyt.utils.misc import wrap_print

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews, not_done=None):
        if not_done is None:
            if self.rewems is None:
                self.rewems = rews
            else:
                self.rewems = self.rewems * self.gamma + rews
            return self.rewems

        else:
            if self.rewems is None:
                self.rewems = rews
            else:
                mask = np.where(not_done == 1.0)
                self.rewems[mask] = self.rewems[mask] * self.gamma + rews[mask]
            return deepcopy(self.rewems)

def generate_observation_stats(env, nsteps=10000):
    '''
    Steps through the environment randomly and produces an observation mean and standard deviation. 
    From https://github.com/openai/large-scale-curiosity/blob/master/utils.py
    '''
    wrap_print('Generating observation mean/std... ({} random steps)'.format(nsteps))
    ob = env.reset()
    obs = []
    for _ in range(nsteps):
        ac = env.action_space.sample()
        ob, _, done, _ = env.step(ac.item())
        if done:
            ob = env.reset()
        ob = np.split(ob, ob.shape[0])
        for o in ob: # stacked observations
            obs.append(o)
    obs = np.array(obs)
    mean = np.mean(obs, 0).astype(np.float32)
    std = np.std(obs, 0).astype(np.float32)
    # print(set(list(np.squeeze(mean).ravel())))
    # cv2.imwrite('norm_images/mean.png', np.squeeze(mean))
    # cv2.imwrite('norm_images/std.png', np.squeeze(std))
    return mean, std



