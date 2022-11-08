from copy import deepcopy
import numpy as np

from rlpyt.agents.base import AgentInputs, IcmAgentCuriosityInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args


class BaseCollector:
    """Class that steps environments, possibly in worker process."""

    def __init__(
            self,
            rank,
            envs,
            samples_np,
            batch_T,
            TrajInfoCls,
            agent=None,  # Present or not, depending on collector class.
            sync=None,
            step_buffer_np=None,
            global_B=1,
            env_ranks=None,
            no_extrinsic=False
            ):
        save__init__args(locals())

    def start_envs(self):
        """e.g. calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        """In CPU-collectors, call ``agent.collector_initialize()`` e.g. to set up
        vector epsilon-greedy, and reset the agent.
        """
        if getattr(self, "agent", None) is not None:  # Not in GPU collectors.
            self.agent.collector_initialize(
                global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
                env_ranks=self.env_ranks,
            )
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, agent_inputs, traj_infos):
        """Main data collection loop."""
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class BaseEvalCollector:
    """Collectors for offline agent evalution; not to record intermediate samples."""

    def __init__(
            self,
            rank,
            envs,
            TrajInfoCls,
            traj_infos_queue,
            max_T,
            agent=None,
            sync=None,
            step_buffer_np=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self):
        """Run agent evaluation in environment and return completed trajectory
        infos."""
        raise NotImplementedError


class DecorrelatingStartCollector(BaseCollector):
    """Collector which can step all environments through a random number of random
    actions during startup, to decorrelate the states in training batches.
    """

    def start_envs(self, max_decorrelation_steps=0):
        """Calls ``reset()`` on every environment instance, then steps each
        one through a random number of random actions, and returns the
        resulting agent_inputs buffer (`observation`, `prev_action`,
        `prev_reward`)."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]

        prev_action = np.stack([env.action_space.null_value() for env in self.envs]) # noop
        prev_reward = np.zeros(len(self.envs), dtype="float32") # total reward (extrinsic + intrinsic)
        observations = list()
        for env in self.envs:
            o = env.reset()
            observations.append(deepcopy(o)) # emulates stepping with noop
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, obs in enumerate(observations):
            observation[b] = obs

        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}")
        if max_decorrelation_steps != 0:
            for b, env in enumerate(self.envs):
                n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
                for _ in range(n_steps):
                    a = env.action_space.sample()
                    if a.shape == (): # 'a' gets stored, but if form is array(3) you need to pass int(3) for env
                        action = int(a)
                    else:
                        action = a
                    o, r, d, info = env.step(action)

                    traj_infos[b].step(o, a, r, d, None, info)
                    if getattr(info, "traj_done", d):
                        o = env.reset()
                        traj_infos[b] = self.TrajInfoCls()
                    if d:
                        a = env.action_space.null_value()
                        r = 0
                observation[b] = o
                prev_action[b] = a
                prev_reward[b] = r

        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.prev_action[:] = prev_action
            self.step_buffer_np.prev_reward[:] = prev_reward
            self.step_buffer_np.observation[:] = observation

        return AgentInputs(observation, prev_action, prev_reward), traj_infos

