
import numpy as np
import gym
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from collections import namedtuple

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import is_namedtuple_class
from rlpyt.envs.wrappers.general_wrappers import *
from rlpyt.envs.wrappers.mario_wrappers import *
from rlpyt.envs.wrappers.pycolab_wrappers import *


class GymEnvWrapper(Wrapper):
    """Gym-style wrapper for converting the Openai Gym interface to the
    rlpyt interface.  Action and observation spaces are wrapped by rlpyt's
    ``GymSpaceWrapper``.

    Output `env_info` is automatically converted from a dictionary to a
    corresponding namedtuple, which the rlpyt sampler expects.  For this to
    work, every key that might appear in the gym environments `env_info` at
    any step must appear at the first step after a reset, as the `env_info`
    entries will have sampler memory pre-allocated for them (so they also
    cannot change dtype or shape).  (see `EnvInfoWrapper`, `build_info_tuples`,
    and `info_to_nt` in file or more help/details)

    Warning:
        Unrecognized keys in `env_info` appearing later during use will be
        silently ignored.

    This wrapper looks for gym's ``TimeLimit`` env wrapper to
    see whether to add the field ``timeout`` to env info.   
    """

    def __init__(self, env, act_null_value=0, obs_null_value=0, force_float32=True):
        super().__init__(env)
        o = self.env.reset()
        o, r, d, info = self.env.step(self.env.action_space.sample())
        env_ = self.env
        time_limit = isinstance(self.env, TimeLimit)
        while not time_limit and hasattr(env_, "env"):
            env_ = env_.env
            time_limit = isinstance(self.env, TimeLimit)
        if time_limit:
            info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
        self._time_limit = time_limit
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        build_info_tuples(info)

    def step(self, action):
        """Reverts the action from rlpyt format to gym format (i.e. if composite-to-
        dictionary spaces), steps the gym environment, converts the observation
        from gym to rlpyt format (i.e. if dict-to-composite), and converts the
        env_info from dictionary into namedtuple."""
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info)
        if isinstance(r, float):
            r = np.dtype("float32").type(r)  # Scalar float32.
        return EnvStep(obs, r, d, info)

    def reset(self):
        """Returns converted observation from gym env reset."""
        return self.observation_space.convert(self.env.reset())

    @property
    def spaces(self):
        """Returns the rlpyt spaces for the wrapped env."""
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )


def build_info_tuples(info, name="info"):
    # Define namedtuples at module level for pickle.
    # Only place rlpyt uses pickle is in the sampler, when getting the
    # first examples, to avoid MKL threading issues...can probably turn
    # that off, (look for subprocess=True --> False), and then might
    # be able to define these directly within the class.
    ntc = globals().get(name)  # Define at module level for pickle.
    info_keys = [str(k).replace(".", "_") for k in info.keys()]
    if ntc is None:
        globals()[name] = namedtuple(name, info_keys)
    elif not (is_namedtuple_class(ntc) and
            sorted(ntc._fields) == sorted(info_keys)):
        raise ValueError(f"Name clash in globals: {name}.")
    for k, v in info.items():
        if isinstance(v, dict):
            build_info_tuples(v, "_".join([name, k]))


def info_to_nt(value, name="info"):
    if not isinstance(value, dict):
        return value
    ntc = globals()[name]
    # Disregard unrecognized keys:
    values = {k: info_to_nt(v, "_".join([name, k]))
        for k, v in value.items() if k in ntc._fields}
    # Can catch some missing values (doesn't nest):
    values.update({k: 0 for k in ntc._fields if k not in values})
    return ntc(**values)


# To use: return a dict of keys and default values which sometimes appear in
# the wrapped env's env_info, so this env always presents those values (i.e.
# make keys and values keep the same structure and shape at all time steps.)
# Here, a dict of kwargs to be fed to `sometimes_info` should be passed as an
# env_kwarg into the `make` function, which should be used as the EnvCls.
# def sometimes_info(*args, **kwargs):
#     # e.g. Feed the env_id.
#     # Return a dictionary (possibly nested) of keys: default_values
#     # for this env.
#     return {}
class EnvInfoWrapper(Wrapper):
    """Gym-style environment wrapper to infill the `env_info` dict of every
    ``step()`` with a pre-defined set of examples, so that `env_info` has
    those fields at every step and they are made available to the algorithm in
    the sampler's batch of data.
    """

    def __init__(self, env, info_example):
        super().__init__(env)
        # self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._sometimes_info = info_example

    def step(self, action):
        """If need be, put extra fields into the `env_info` dict returned.
        See file for function ``infill_info()`` for details."""
        o, r, d, info = super().step(action)
        # Try to make info dict same key structure at every step.
        return o, r, d, infill_info(info, self._sometimes_info)


def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info


def make(*args, info_example=None, **kwargs):
    """Use as factory function for making instances of classic gym environments with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``.  If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    env = gym.make(kwargs['id'])
    if kwargs['no_extrinsic']:
        env = NoExtrinsicReward(env)
    if kwargs['no_negative_reward']:
        env = NoNegativeReward(env)
    if info_example is None:
        env = GymEnvWrapper(env)
    else:
        env = GymEnvWrapper(EnvInfoWrapper(env, info_example)) 

    return env

def mario_make(*args, info_example=None, **kwargs):
    """Use as factory function for making instances of SuperMario environments with
    rlpyt's ``GymEnvWrapper``, using ``gym_super_mario_bros.make(*args, **kwargs)``. If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    import retro
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

    env = gym_super_mario_bros.make(kwargs['game'])
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    if kwargs['no_negative_reward']:
        env = NoNegativeReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=True)
    env = FrameStack(env, 4)
    env = PytorchImage(env) # (h,w,c) -> (c,h,w)
    if info_example is None:
        env = GymEnvWrapper(env)
    else:
        env = GymEnvWrapper(EnvInfoWrapper(env))
    return env

def deepmind_make(*args, info_example=None, **kwargs):
    """Use as factory function for making instances of Pycolab environments with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``. If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    import rlpyt.envs.mazeworld.mazeworld

    env = gym.make(kwargs['game'], 
                   obs_type=kwargs['obs_type'], 
                   max_iterations=kwargs['max_steps_per_episode'])
    env.heatmap_init(kwargs['logdir'], kwargs['log_heatmaps'])
    
    if kwargs['obs_type'] == 'rgb_full':
        resize_scale = 84 // env.width
        if env.width == 85:
            resize_scale = 1
    else:
        resize_scale = 17
    env.obs_init(resize_scale)

    if kwargs['no_negative_reward']:
        env = NoNegativeReward(env)

    if 'rgb' in kwargs['obs_type']:
        if kwargs['grayscale'] == True:
            env = Grayscale(env)
        if kwargs['obs_type'] == 'rgb_full' and env.width != env.height:
            env = ResizeFull(env)
        env = PytorchImage(env)

    if info_example is None:
        env = GymEnvWrapper(env, act_null_value=env.act_null_value)
    else:
        env = GymEnvWrapper(EnvInfoWrapper(env))

    return env


