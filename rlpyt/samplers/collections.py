import numpy as np
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict


Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
    ["action", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
    ["action", "prev_action", "agent_info", "bootstrap_value", "bootstrap_value_int", "bootstrap_value_ext_prime"])
EnvSamples = namedarraytuple("EnvSamples",
    ["reward", "prev_reward", "observation","next_observation", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """
    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.EpExtrinsicReward = 0
        self.EpNonzeroExtrinsicRewards = 0
        self.EpDiscountedExtrinsicReward = 0
        self._cur_discount = 1
        self.EpAveExtrinsicReward = []

    def step(self, observation, action, reward_ext, done, agent_info, env_info):
        self.Length += 1
        self.EpExtrinsicReward += reward_ext
        self.EpNonzeroExtrinsicRewards += reward_ext != 0
        self.EpDiscountedExtrinsicReward += self._cur_discount * reward_ext
        self._cur_discount *= self._discount
        self.EpAveExtrinsicReward.append(reward_ext)

    def terminate(self):
        self.EpAveExtrinsicReward = np.mean(self.EpAveExtrinsicReward)
        return self


















