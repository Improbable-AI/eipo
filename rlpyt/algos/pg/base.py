
import numpy as np
import torch
from collections import namedtuple
from scipy.stats import linregress

from rlpyt.algos.base import RlAlgorithm
from rlpyt.algos.utils import discount_return, generalized_advantage_estimation, valid_from_done

# Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase
OptInfo = namedtuple("OptInfo", ["return_",
                                 "return_int_",
                                 "intrinsic_rewards",
                                 "normalized_extreward",
                                 "normalized_intreward",
                                 "rescaled_extreward",
                                 "rescaled_intreward",
                                 "advantage_ext",
                                 "advantage_int",
                                 "advantage_int_weighted",
                                 "value_ext",
                                 "extint_ratio",
                                 "valpred",
                                 "advantage",
                                 "loss",
                                 "pi_loss",
                                 "pi_ext_loss",
                                 "pi_int_loss",
                                 "dual_policy_weights",
                                 "kl_scores",
                                 "value_loss",
                                 "value_int_loss",
                                 "entropy_loss",
                                 "entropy_ext_loss",
                                 "entropy_int_loss",
                                 "inv_loss",
                                 "forward_loss",
                                 "reward_total_std",
                                 "curiosity_loss",
                                 "entropy",
                                 "entropy_ext",
                                 "entropy_int",
                                 "perplexity",
                                 "perplexity_ext",
                                 "perplexity_int",
                                 "utility_nw",
                                 "kl_constraint",

                                 # For minmax
                                 "value",
                                 "value_int",
                                 "value_ext_prime",

                                 "loss_max",
                                 "pi_max_loss",
                                 "value_max_ext_prime_loss",
                                 "entropy_max_loss",
                                 "entropy_max",
                                 "perplexity_max",
                                 "advantage_max",

                                 "loss_min",
                                 "pi_min_loss",
                                 "value_min_loss",
                                 "value_min_int_loss",
                                 "entropy_min_loss",
                                 "entropy_min",
                                 "perplexity_min",
                                 "advantage_min",

                                 "max_steps",
                                 "min_steps",

                                 "minmax_alpha",
                                 "alpha_loss",
                                 ])


AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])


class PolicyGradientAlgo(RlAlgorithm):
    """
    Base policy gradient / actor-critic algorithm, which includes
    initialization procedure and processing of data samples to compute
    advantages.
    """

    bootstrap_value = True  # Tells the sampler it needs Value(State')
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset

    def process_returns(self, samples, dual_value=False, dual_policy_weights=None):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.
        """
        reward, done, value, value_int, bv, bv_int = (samples.env.reward,
                                                      samples.env.done,
                                                      samples.agent.agent_info.value,
                                                      samples.agent.agent_info.value_int,
                                                      samples.agent.bootstrap_value,
                                                      samples.agent.bootstrap_value_int)
        done = done.type(reward.dtype)
        not_done = np.abs(done-1)

        if self.normalize_extreward:
            reward_copy = reward.clone().data.numpy()
            rews = np.array([self.extreward_ff.update(reward_copy[i], not_done=not_done[i]) for i in range(len(reward_copy))])
            self.extreward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
            reward = reward / np.sqrt(self.extreward_rms.var)
            self.normalized_extreward = reward.clone().data.numpy()

        if self.rescale_extreward:
            reward_copy = reward.clone().data.numpy()
            reward = (reward-np.min(reward_copy))/(np.max(reward_copy)-np.min(reward_copy)+1e-15)
            self.rescaled_extreward = reward.clone().data.numpy()

        intrinsic = False
        if self.curiosity_type in {'icm', 'disagreement', 'micm'}:
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.env.next_observation.clone(), samples.agent.action.clone())
            intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
            self.intrinsic_rewards = intrinsic_rewards_logging
            self.extint_ratio = reward.clone().data.numpy()/(intrinsic_rewards_logging+1e-15)
            intrinsic = True
        elif self.curiosity_type == 'ndigo':
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.agent.prev_action.clone(), samples.agent.action.clone()) # no grad
            intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
            self.intrinsic_rewards = intrinsic_rewards_logging
            self.extint_ratio = reward.clone().data.numpy()/(intrinsic_rewards_logging+1e-15)
            intrinsic = True
        elif self.curiosity_type == 'rnd':
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.next_observation.clone(), done.clone())
            intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
            self.intrinsic_rewards = intrinsic_rewards_logging
            self.extint_ratio = reward.clone().data.numpy()/(intrinsic_rewards_logging+1e-15)
            intrinsic = True

        if intrinsic:
            if self.normalize_intreward:
                intrinsic_rewards_copy = intrinsic_rewards.clone().data.numpy()
                rews = np.array([self.intreward_ff.update(intrinsic_rewards_copy[i], not_done=not_done[i]) for i in range(len(intrinsic_rewards_copy))])
                self.intreward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
                intrinsic_rewards = intrinsic_rewards / np.sqrt(self.intreward_rms.var)
                self.normalized_intreward = intrinsic_rewards.clone().data.numpy()

            if self.rescale_intreward:
                reward_copy = reward.clone().data.numpy()
                intrinsic_rewards_copy = intrinsic_rewards.clone().data.numpy()
                intrinsic_rewards = (intrinsic_rewards-np.min(intrinsic_rewards_copy))/(np.max(intrinsic_rewards_copy)-np.min(intrinsic_rewards_copy))
                delta = np.mean(reward_copy)/np.mean(intrinsic_rewards.data.numpy())
                intrinsic_rewards *= delta
                self.rescaled_intreward = intrinsic_rewards.clone().data.numpy()

        if not dual_value:
            if intrinsic:
                reward += intrinsic_rewards

            if self.normalize_reward:
                reward_copy = reward.clone().data.numpy()
                rews = np.array([self.reward_ff.update(reward_copy[i], not_done=not_done[i]) for i in range(len(reward_copy))])
                self.reward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
                reward = reward / np.sqrt(self.reward_rms.var)

        return_int_ = torch.zeros(reward.shape) # placeholder
        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            if dual_value and intrinsic:
                return_int_ = discount_return(intrinsic_rewards, done, bv_int, self.discount_ri)
                advantage_int = return_int_ - value_int
                advantage_ext = return_ - value
                if dual_policy_weights is not None:
                    self.advantage_int_weighted = (dual_policy_weights*advantage_int.clone().detach()).data.numpy()
                    advantage = advantage_ext + (dual_policy_weights*advantage_int)
                else:
                    advantage = advantage_ext + advantage_int
            else:
                advantage = return_ - value
                advantage_ext = None
                advantage_int = None
        else:
            if dual_value and intrinsic:
                advantage_ext, return_ = generalized_advantage_estimation(reward, value, done, bv, self.discount, self.gae_lambda)
                advantage_int, return_int_ = generalized_advantage_estimation(intrinsic_rewards, value_int, done, bv_int, self.discount_ri, self.gae_lambda)
                if dual_policy_weights is not None:
                    self.advantage_int_weighted = (dual_policy_weights*advantage_int.clone().detach()).data.numpy()
                    advantage = advantage_ext + (dual_policy_weights*advantage_int)
                else:
                    advantage = advantage_ext + advantage_int
            else:
                advantage, return_ = generalized_advantage_estimation(reward, value, done, bv, self.discount, self.gae_lambda)
                advantage_ext = None
                advantage_int = None


        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, return_int_, advantage, advantage_ext, advantage_int, valid


    def _process_intrinsic_rewards(self, samples):
      reward = samples.env.reward
      done = samples.env.done
      done = done.type(reward.dtype)
      not_done = np.abs(done-1)
      if self.curiosity_type in {'icm', 'disagreement', 'micm'}:
          intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.env.next_observation.clone(), samples.agent.action.clone())
          if self.normalize_intreward:
            intrinsic_rewards_copy = intrinsic_rewards.clone().data.numpy()
            rews = np.array([self.intreward_ff.update(intrinsic_rewards_copy[i], not_done=not_done[i]) for i in range(len(intrinsic_rewards_copy))])
            self.intreward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
            intrinsic_rewards = intrinsic_rewards / np.sqrt(self.intreward_rms.var)
            self.normalized_intreward = intrinsic_rewards.clone().data.numpy()
          intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
          self.intrinsic_rewards = intrinsic_rewards_logging
      elif self.curiosity_type == 'ndigo':
          intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.agent.prev_action.clone(), samples.agent.action.clone()) # no grad
          intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
          self.intrinsic_rewards = intrinsic_rewards_logging
      elif self.curiosity_type == 'rnd':
          intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.next_observation.clone(), done.clone())
          intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
          self.intrinsic_rewards = intrinsic_rewards_logging
      else:
        raise NotImplemented()
      return intrinsic_rewards, intrinsic_rewards_logging

    def normalize_extreward(self, samples):
      reward, done = samples.env.reward, samples.env.done
      done = done.type(reward.dtype)
      not_done = np.abs(done - 1)
      reward_copy = reward.clone().data.numpy()
      rews = np.array([self.extreward_ff.update(reward_copy[i], not_done=not_done[i]) for i in range(len(reward_copy))])
      self.extreward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
      reward = reward / np.sqrt(self.extreward_rms.var)
      self.normalized_extreward = reward.clone().data.numpy()

      return reward

    def normalize_reward(self, samples):
        reward, done = samples.env.reward, samples.env.done
        done = done.type(reward.dtype)
        not_done = np.abs(done - 1)
        reward_copy = reward.clone().data.numpy()
        rews = np.array([self.reward_ff.update(reward_copy[i], not_done=not_done[i]) for i in range(len(reward_copy))])
        self.reward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
        reward = reward / np.sqrt(self.reward_rms.var)
        return reward

    def process_returns_min_step(self, samples, reward, intrinsic_rewards, alpha, dual_value):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.

        params: alpha: Largarangian multiplier
        """
        done, value, bv, value_int, bv_int = (
                                  samples.env.done,
                                  samples.agent.agent_info.value, # V^pi_E
                                  samples.agent.bootstrap_value, # V^pi_E
                                  samples.agent.agent_info.value_int, # V^pi_I
                                  samples.agent.bootstrap_value_int, # V^pi_I
                                  )
        done = done.type(reward.dtype)
        not_done = np.abs(done-1)



        # Prepare for some quantities for value calculation
        # if not dual_value:
        #     r_ = reward + intrinsic_rewards # For value function
        #     v_ = value # V^pi_{E+I}(s_t)
        #     bv_ = bv # V^pi_{E+I}(s_t+1)
        # else:
        #     r_ = reward # For value function
        #     v_ = value + value_int # V^pi_{E+I}(s_t)
        #     bv_ = bv + bv_int # V^pi_{E+I}(s_t+1)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            raise NotImplemented()
            # For value function approximation
            return_ = discount_return(reward, done, bv, self.discount)
            return_int_ = discount_return(intrinsic_rewards, done, bv_int, self.discount_ri)
            # For U_-(s,a)
            return_u_ = discount_return(alpha * reward, done, bv + bv_int, self.discount)
            advantage = return_u_ - (value + value_int)
            # advantage_pi = (return_ + return_int_) - (value + value_int)
        else:
            # For value function approximation an
            advantage_ext, return_ = generalized_advantage_estimation(reward, value, done, bv, self.discount, self.gae_lambda)
            advantage_int, return_int_ = generalized_advantage_estimation(intrinsic_rewards, value_int, done, bv_int, self.discount_ri, self.gae_lambda)
            # Compute A_E and A_I for A_{E+I}
            advantage_extint = advantage_ext + advantage_int
            # For U_-(s,a) = (alpha - 1) * r^E - r^I + A_{E+I}(s,a)
            advantage = (alpha - 1) * reward - intrinsic_rewards + advantage_extint
            advantage_pi = advantage_extint

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, return_int_, advantage, advantage_pi, valid

    def process_returns_max_step(self, samples, reward, intrinsic_rewards, alpha):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.

        params: alpha: Largarangian multiplier
        """
        done, value_ext_prime, value_int, bv_ext_prime, bv_int = (
                                  samples.env.done,
                                  samples.agent.agent_info.value_ext_prime,
                                  samples.agent.agent_info.value_int,
                                  samples.agent.bootstrap_value_ext_prime,
                                  samples.agent.bootstrap_value_int
                                  )
        done = done.type(reward.dtype)
        not_done = np.abs(done-1)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            raise NotImplemented()
            # For value function approximation
            return_ = discount_return(reward, done, bv_ext_prime, self.discount)
            return_int_ = discount_return(intrinsic_rewards, done, bv_int, self.discount_ri)
            # For U_+(s,a)
            return_u_ = discount_return((1 + alpha) * reward + intrinsic_rewards, done, alpha * bv_ext_prime, self.discount)
            advantage = return_u_ - alpha * value_ext_prime
            advantage_pi_prime = return_ - value_ext_prime
        else:
            # For U_+(s,a) = r^E + r^I + alpha A_E(s_t, a_t)
            advantage_ext, return_ = generalized_advantage_estimation(reward, value_ext_prime, done, bv_ext_prime, self.discount, self.gae_lambda)
            _, return_int_ = generalized_advantage_estimation(intrinsic_rewards, value_int, done, bv_int, self.discount_ri, self.gae_lambda)
            advantage = reward + intrinsic_rewards + alpha * advantage_ext
            advantage_pi_prime = advantage_ext

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, return_int_, advantage, advantage_pi_prime, valid