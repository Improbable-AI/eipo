
import torch

from rlpyt.agents.base import AgentStep, AgentCuriosityStep, BaseAgent, RecurrentAgentMixin, AlternatingRecurrentAgentMixin
from rlpyt.agents.pg.base import AgentInfo, NdigoInfo, IcmInfo, RndInfo, AgentInfoRnn
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method


class CategoricalPgAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GaussianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        pi, value = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value = self.model(*model_inputs)
        return value.to("cpu")


class RecurrentCategoricalPgAgentBase(BaseAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state), device=self.device)
        pi, pi_ext, pi_int, value, value_int, value_ext_prime, next_rnn_state = self.model(*model_inputs)
        dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime = buffer_to((DistInfo(prob=pi), DistInfo(prob=pi_ext), DistInfo(prob=pi_int), value, value_int, value_ext_prime), device="cpu")
        return dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime, next_rnn_state  # Leave rnn_state on device.

    def initialize(self, env_spaces, share_memory=False, global_B=1, obs_stats=None, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B=global_B, obs_stats=obs_stats, env_ranks=env_ranks)
        self.distribution = Categorical(dim=env_spaces.action.n)
        if self.dual_policy != 'default':
            self.distribution_ext = Categorical(dim=env_spaces.action.n)
            self.distribution_int = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        pi, pi_ext, pi_int, value, value_int, value_ext_prime, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        dist_info = DistInfo(prob=pi)
        dist_ext_info = DistInfo(prob=pi_ext)
        dist_int_info = DistInfo(prob=pi_int)
        if self.use_minmax:
          action = self.distribution.sample(dist_ext_info) if self.is_max_step else self.distribution.sample(dist_info)
        else:
          if self.dual_policy in {'combined', 'default'}:
              action = self.distribution.sample(dist_info)
          elif self.dual_policy in {'ext'}:
              action = self.distribution_ext.sample(dist_ext_info)
          elif self.dual_policy in {'int'}:
              action = self.distribution_int.sample(dist_int_info)
        # Model handles None, but Buffer does not, make zeros if needed:
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, dist_ext_info=dist_ext_info, dist_int_info=dist_int_info, value=value, value_int=value_int, value_ext_prime=value_ext_prime, prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def curiosity_step(self, curiosity_type, *args):

        if curiosity_type in {'icm', 'micm', 'disagreement'}:
            observation, next_observation, actions = args
            actions = self.distribution.to_onehot(actions)
            curiosity_agent_inputs = buffer_to((observation, next_observation, actions), device=self.device)
            agent_curiosity_info = IcmInfo()
        elif curiosity_type == 'ndigo':
            observation, prev_actions, actions = args
            actions = self.distribution.to_onehot(actions)
            prev_actions = self.distribution.to_onehot(prev_actions)
            curiosity_agent_inputs = buffer_to((observation, prev_actions, actions), device=self.device)
            agent_curiosity_info = NdigoInfo(prev_gru_state=None)
        elif curiosity_type == 'rnd':
            next_observation, done = args
            curiosity_agent_inputs = buffer_to((next_observation, done), device=self.device)
            agent_curiosity_info = RndInfo()

        r_int = self.model.curiosity_model.compute_bonus(*curiosity_agent_inputs)
        r_int, agent_curiosity_info = buffer_to((r_int, agent_curiosity_info), device="cpu")
        return AgentCuriosityStep(r_int=r_int, agent_curiosity_info=agent_curiosity_info)

    def curiosity_loss(self, curiosity_type, *args):

        if curiosity_type in {'icm', 'micm'}:
            observation, next_observation, actions, valid = args
            actions = self.distribution.to_onehot(actions)
            actions = actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            curiosity_agent_inputs = buffer_to((observation, next_observation, actions, valid), device=self.device)
            inv_loss, forward_loss = self.model.curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (inv_loss.to("cpu"), forward_loss.to("cpu"))
        elif curiosity_type == 'disagreement':
            observation, next_observation, actions, valid = args
            actions = self.distribution.to_onehot(actions)
            actions = actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            curiosity_agent_inputs = buffer_to((observation, next_observation, actions, valid), device=self.device)
            forward_loss = self.model.curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (forward_loss.to("cpu"))
        elif curiosity_type == 'ndigo':
            observations, prev_actions, actions, valid = args
            actions = self.distribution.to_onehot(actions)
            prev_actions = self.distribution.to_onehot(prev_actions)
            actions = actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            prev_actions = prev_actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            curiosity_agent_inputs = buffer_to((observations, prev_actions, actions, valid), device=self.device)
            forward_loss = self.model.curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (forward_loss.to("cpu"))
        elif curiosity_type == 'rnd':
            next_observation, valid = args
            curiosity_agent_inputs = buffer_to((next_observation, valid), device=self.device)
            forward_loss = self.model.curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (forward_loss.to("cpu"))

        return losses

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _pi_ext, _pi_int, value, value_int, value_ext_prime, _rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        return value.to("cpu"), value_int.to("cpu"), value_ext_prime.to("cpu")


class RecurrentCategoricalPgAgent(RecurrentAgentMixin, RecurrentCategoricalPgAgentBase):
    pass


class AlternatingRecurrentCategoricalPgAgent(AlternatingRecurrentAgentMixin, RecurrentCategoricalPgAgentBase):
    pass


