
import numpy as np
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, IcmAgentCuriosityInputs, NdigoAgentCuriosityInputs, RndAgentCuriosityInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.grad_utils import plot_grad_flow

LossInputs = namedarraytuple("LossInputs", ["agent_inputs",
                                            "agent_curiosity_inputs",
                                            "action",
                                            "return_",
                                            "return_int_",
                                            "advantage",
                                            "advantage_ext",
                                            "advantage_int",
                                            "valid",
                                            "old_dist_info",
                                            "old_dist_ext_info",
                                            "old_dist_int_info"])

LossMinStepInputs = namedarraytuple("LossMinStepInputs", [
                                            "agent_inputs",
                                            "action",
                                            "return_",
                                            "return_int_",
                                            "advantage",
                                            "advantage_pi",
                                            "advantage_pi_prime",
                                            "valid",
                                            "old_dist_info",])

LossMaxStepInputs = namedarraytuple("LossMaxStepInputs", [
                                            "agent_inputs",
                                            "action",
                                            "return_",
                                            "advantage",
                                            "advantage_pi",
                                            "advantage_pi_prime",
                                            "valid",
                                            "old_dist_ext_info",])


LossAlphaInputs = namedarraytuple("LossAlphaInputs", [
                                            "agent_inputs",
                                            "action",
                                            "advantage_pi_prime",
                                            "valid",
                                            "old_dist_ext_info",])


class PPO(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """

    def __init__(
            self,
            discount=0.99,
            discount_ri=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_reward=False,
            normalize_extreward=False,
            normalize_intreward=False,
            rescale_extreward=False,
            rescale_intreward=False,
            dual_value=False,
            dual_policy='default',
            dual_policy_noint=False,
            dual_policy_weighting='none',
            dpw_formulation='inverse',
            utility_noworkers=False,
            kl_lambda=1.0,
            kl_clamp=0.0,
            util_clamp=0.2,
            util_detach='none',
            kl_detach='none',
            importance_sample=0.,
            curiosity_type='none',

            # Minmax
            use_minmax=False,
            minmax_alpha=0.1,
            use_adapt_alpha=False,
            alpha_lr=0.001,
            alpha_clip="none",
            minmax_ablation='none',
            minmax_switch='none',
            initial_algo_state_dict=None,
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        if self.normalize_reward:
            self.reward_ff = RewardForwardFilter(discount)
            self.reward_rms = RunningMeanStd()
        if self.normalize_extreward:
            self.extreward_ff = RewardForwardFilter(discount)
            self.extreward_rms = RunningMeanStd()
        if self.normalize_intreward:
            self.intreward_ff = RewardForwardFilter(discount)
            self.intreward_rms = RunningMeanStd()
        if self.dual_policy_weighting != 'none':
            self.advantage_int_weighted = np.array(0.0)
        self.intrinsic_rewards = None
        self.normalized_extreward = None
        self.normalized_intreward = None
        self.rescaled_extreward = None
        self.rescaled_intreward = None
        self.extint_ratio = None

        # rand-low,high
        if minmax_alpha.startswith("rand"):
            low, high = minmax_alpha.split('_')[1].split(',')
            self.init_minmax_alpha = self.minmax_alpha = np.random.uniform(low=float(low), high=float(high))
        else:
            self.init_minmax_alpha = self.minmax_alpha = float(minmax_alpha)
        self.minmax_stage_counter = 0 # will circulate between 0 and 1
        if alpha_clip == 'none':
          self.alpha_clip = None
          self.clip_alpha_grad = False
        elif alpha_clip.startswith('g'):
          lb, ub = alpha_clip[1:].split(",")
          self.alpha_clip = (float(lb), float(ub))
          self.clip_alpha_grad = True
        else:
          lb, ub = alpha_clip.split(",")
          self.alpha_clip = (float(lb), float(ub))
          self.clip_alpha_grad = False

        self.need_flip = False
        self.last_max_step_advantage = 0
        self.last_min_step_advantage = 0

    def algo_state_dict(self):
        return {
          "alpha": self.minmax_alpha,
          "init_minmax_alpha": self.init_minmax_alpha,
          "last_max_step_advantage": self.last_max_step_advantage,
          "last_min_step_advantage": self.last_min_step_advantage
        }

    def load_algo_state_dict(self, algo_state_dict):
        self.minmax_alpha = algo_state_dict["alpha"]
        self.init_minmax_alpha = algo_state_dict["init_minmax_alpha"]
        self.last_max_step_advantage = algo_state_dict["last_max_step_advantage"]
        self.last_min_step_advantage = algo_state_dict["last_min_step_advantage"]

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        if self.initial_algo_state_dict is not None:
            self.load_algo_state_dict(self.initial_algo_state_dict)

        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

        if self.minmax_ablation == 'decouple':
          self.agent.is_max_step = False # we always run min_step in decouple

    def _get_curiosity_agent_input(self, samples, valid):
        if self.curiosity_type in {'icm', 'micm', 'disagreement'}:
            agent_curiosity_inputs = IcmAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                next_observation=samples.env.next_observation.clone(),
                action=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'ndigo':
            agent_curiosity_inputs = NdigoAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                prev_actions=samples.agent.prev_action.clone(),
                actions=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'rnd':
            agent_curiosity_inputs = RndAgentCuriosityInputs(
                next_observation=samples.env.next_observation.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'none':
            agent_curiosity_inputs = None

        return agent_curiosity_inputs

    def _optimize_agent_minmax(self, itr, samples):
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        init_rnn_state = None
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)

        # Compute for max step and min step

        # We need intrinsic rewards to do minmax
        assert self.curiosity_type in {'icm', 'micm', 'disagreement', 'ndigo', 'rnd'}
        intrinsic_rewards, intrinsic_rewards_logging = self._process_intrinsic_rewards(samples)
        if self.normalize_extreward:
          reward = self.normalize_extreward(samples)
        elif self.normalize_reward:
          reward = self.normalize_reward(samples)
        else:
          reward = samples.env.reward
        return_max_, return_max_int_, advantage_max, advantage_pi_prime, valid = self.process_returns_max_step(samples, reward, intrinsic_rewards, self.minmax_alpha)
        return_min_, return_min_int_, advantage_min, advantage_pi, valid = self.process_returns_min_step(samples, reward, intrinsic_rewards, self.minmax_alpha, dual_value=self.dual_value)

        agent_curiosity_inputs = self._get_curiosity_agent_input(samples, valid)

        loss_max_inputs = LossMaxStepInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_max_,
            advantage=advantage_max,
            advantage_pi_prime=advantage_pi_prime,
            advantage_pi=advantage_pi,
            valid=valid,
            old_dist_ext_info=samples.agent.agent_info.dist_ext_info,
        )

        loss_min_inputs = LossMinStepInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_min_,
            return_int_=return_min_int_,
            advantage=advantage_min,
            advantage_pi_prime=advantage_pi_prime,
            advantage_pi=advantage_pi,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )

        loss_alpha_inputs = LossAlphaInputs(agent_inputs=agent_inputs,
                    action=samples.agent.action,
                    advantage_pi_prime=advantage_pi_prime,
                    valid=valid,
                    old_dist_ext_info=samples.agent.agent_info.dist_ext_info)

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        T, B = samples.env.reward.shape[:2]

        opt_info =  OptInfo(*([] for _ in range(len( OptInfo._fields))))

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        self.minmax_stage_counter += 1

        for epoch in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None

                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                # We do min and max both at an iteration for the logging purpose (one will be no_grad)
                if self.agent.is_max_step:
                  ret_alpha_loss = (epoch == self.epochs - 1) & self.use_adapt_alpha
                  loss_max, pi_max_loss, value_max_ext_prime_loss, entropy_max_loss, entropy_max, perplexity_max, alpha_loss = self.loss_max_step(*loss_max_inputs[T_idxs, B_idxs], rnn_state,
                                ret_alpha_loss=ret_alpha_loss)
                  if ret_alpha_loss:
                      # TODO: this is a workaround. we take the last epoch's alpha_loss to update alpha, but in theory we should take the alpha loss after we finish this gradient step.
                      alpha_loss = alpha_loss.item()
                      if self.alpha_clip:
                        if self.clip_alpha_grad:
                          alpha_grad = np.clip(alpha_loss, self.alpha_clip[0], self.alpha_clip[1])
                          self.minmax_alpha = self.minmax_alpha - self.alpha_lr * alpha_grad
                        else:
                          self.minmax_alpha = self.minmax_alpha - self.alpha_lr * alpha_loss
                          self.minmax_alpha = np.clip(self.minmax_alpha, self.alpha_clip[0], self.alpha_clip[1])
                  loss = loss_max

                  opt_info.loss_max.append(loss_max.item())
                  opt_info.pi_max_loss.append(pi_max_loss.item())
                  opt_info.value_max_ext_prime_loss.append(value_max_ext_prime_loss.item())
                  opt_info.entropy_max_loss.append(entropy_max_loss.item())
                  opt_info.entropy_max.append(entropy_max.item())
                  opt_info.perplexity_max.append(perplexity_max.item())
                else:
                  loss_min, pi_min_loss, value_min_loss, value_min_int_loss, entropy_min_loss, entropy_min, perplexity_min = self.loss_min_step(*loss_min_inputs[T_idxs, B_idxs], rnn_state)
                  loss = loss_min
                  alpha_loss = 0

                  opt_info.loss_min.append(loss_min.item())
                  opt_info.pi_min_loss.append(pi_min_loss.item())
                  opt_info.value_min_loss.append(value_min_loss.item())
                  opt_info.value_min_int_loss.append(value_min_int_loss.item())
                  opt_info.entropy_min_loss.append(entropy_min_loss.item())
                  opt_info.entropy_min.append(entropy_min.item())
                  opt_info.perplexity_min.append(perplexity_min.item())

                opt_info.alpha_loss.append(alpha_loss)
                opt_info.minmax_alpha.append(self.minmax_alpha)

                curiosity_losses = self._curiosity_losses(agent_curiosity_inputs)
                for curiosity_loss in curiosity_losses:
                  loss += curiosity_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                intrinsic = False
                if self.curiosity_type in {'icm', 'micm'}:
                    inv_loss, forward_loss = curiosity_losses
                    opt_info.inv_loss.append(inv_loss.item())
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'disagreement':
                    forward_loss = curiosity_losses[0]
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'ndigo':
                    forward_loss = curiosity_losses[0]
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'rnd':
                    forward_loss = curiosity_losses[0]
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                if intrinsic:
                    opt_info.intrinsic_rewards.append(self.intrinsic_rewards.flatten())

                if self.normalize_extreward:
                    opt_info.normalized_extreward.append(self.normalized_extreward.flatten())
                if self.normalize_intreward:
                    opt_info.normalized_intreward.append(self.normalized_intreward.flatten())
                if self.rescale_extreward:
                    opt_info.rescaled_extreward.append(self.rescaled_extreward.flatten())
                if self.rescale_intreward:
                    opt_info.rescaled_intreward.append(self.rescaled_intreward.flatten())

                self.update_counter += 1

        # NOTE: computing alpha loss after max-step is correct but slow -- we use the alpha loss of the last epoch to approximate
        # % 2 == 0 means that we just finished the max step
        # if self.minmax_stage_counter % 2 == 0:
        #     with torch.no_grad():
        #         alpha_grad = 0
        #         for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
        #             T_idxs = slice(None) if recurrent else idxs % T
        #             B_idxs = idxs if recurrent else idxs // T
        #             rnn_state = init_rnn_state[B_idxs] if recurrent else None

        #             alpha_grad += self.loss_alpha(*loss_alpha_inputs[T_idxs, B_idxs], rnn_state).cpu().item()
        #     self.minmax_alpha = self.minmax_alpha - self.alpha_lr * alpha_grad

        opt_info.return_.append(torch.mean(return_min_.detach()).detach().clone().item())
        opt_info.return_int_.append(torch.mean(return_max_int_.detach()).detach().clone().item())

        opt_info.advantage_min.append(torch.mean(advantage_min.detach()).detach().clone().item())
        opt_info.advantage_max.append(torch.mean(advantage_max.detach()).detach().clone().item())

        opt_info.value.append(torch.mean(samples.agent.agent_info.value.detach()).detach().clone().item())
        opt_info.value_int.append(torch.mean(samples.agent.agent_info.value_int.detach()).detach().clone().item())
        opt_info.value_ext_prime.append(torch.mean(samples.agent.agent_info.value_ext_prime.detach()).detach().clone().item())

        opt_info.max_steps.append(int(self.agent.is_max_step))
        opt_info.min_steps.append(int(not self.agent.is_max_step))

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        layer_info = dict() # empty dict to store model layer weights for tensorboard visualizations

        # Flip the is_max_step
        if self.minmax_switch == 'win-lose':
          self.need_flip = False
          if self.agent.is_max_step:
            self.need_flip = advantage_max.mean() < 0
          else:
            self.need_flip = advantage_min.mean() < 0
          if self.need_flip:
            self.agent.flip_is_max_step()
        elif self.minmax_switch == 'diff':
          mean_adv_max = advantage_max.mean()
          mean_adv_min = advantage_min.mean()
          if self.agent.is_max_step and (mean_adv_max - self.last_max_step_advantage <= 0):
            self.agent.flip_is_max_step()
          elif (not self.agent.is_max_step) and (mean_adv_min - self.last_min_step_advantage <= 0):
            self.agent.flip_is_max_step()
          self.last_max_step_advantage = mean_adv_max
          self.last_min_step_advantage = mean_adv_min
        elif self.minmax_ablation == 'decouple':
          self.agent.is_max_step = False # Ensure it is min-step always
        else:
          self.agent.flip_is_max_step()

        return opt_info, layer_info

    def _optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        init_rnn_state = None
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        dual_policy_weights = None
        if self.dual_policy_weighting != 'none':
            if init_rnn_state is not None:
                # [B,N,H] --> [N,B,H] (for cudnn).
                init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
                init_rnn_state = buffer_method(init_rnn_state, "contiguous")
                with torch.no_grad():
                    dist_info, dist_ext_info, dist_int_info, value, value_int, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
            else:
                with torch.no_grad():
                    dist_info, dist_ext_info, dist_int_info, value, value_int = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here
            with torch.no_grad():
                dist_og = self.agent.distribution
                if self.dual_policy_weighting == 'ext_first':
                    kl_scores = dist_og.kl(dist_ext_info, dist_int_info)
                    if self.dpw_formulation == 'inverse':
                        dual_policy_weights = 1/torch.clamp(kl_scores, min=0.00000000001, max=100000.0)
                    elif self.dpw_formulation == 'exp':
                        dual_policy_weights = torch.exp(-kl_scores)
                if self.dual_policy_weighting == 'int_first':
                    kl_scores = dist_og.kl(dist_int_info, dist_ext_info)
                    if self.dpw_formulation == 'inverse':
                        dual_policy_weights = 1/torch.clamp(kl_scores, min=0.00000000001, max=100000.0)
                    elif self.dpw_formulation == 'exp':
                        dual_policy_weights = torch.exp(-kl_scores)

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)

        return_, return_int_, advantage, advantage_ext, advantage_int, valid = self.process_returns(samples, self.dual_value, dual_policy_weights)

        if self.curiosity_type in {'icm', 'micm', 'disagreement'}:
            agent_curiosity_inputs = IcmAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                next_observation=samples.env.next_observation.clone(),
                action=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'ndigo':
            agent_curiosity_inputs = NdigoAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                prev_actions=samples.agent.prev_action.clone(),
                actions=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'rnd':
            agent_curiosity_inputs = RndAgentCuriosityInputs(
                next_observation=samples.env.next_observation.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'none':
            agent_curiosity_inputs = None
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            agent_curiosity_inputs=agent_curiosity_inputs,
            action=samples.agent.action,
            return_=return_,
            return_int_=return_int_,
            advantage=advantage,
            advantage_ext=advantage_ext,
            advantage_int=advantage_int,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
            old_dist_ext_info=samples.agent.agent_info.dist_ext_info,
            old_dist_int_info=samples.agent.agent_info.dist_int_info
        )

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        T, B = samples.env.reward.shape[:2]

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None

                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, pi_loss, pi_ext_loss, pi_int_loss, value_loss, value_int_loss, entropy_loss, entropy_ext_loss, entropy_int_loss, entropy, entropy_ext, entropy_int, perplexity, perplexity_ext, perplexity_int, curiosity_losses, utility_nw, kl_constraint = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)

                loss.backward()
                count = 0
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                # Tensorboard summaries
                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.item())
                opt_info.pi_ext_loss.append(pi_ext_loss.item())
                opt_info.pi_int_loss.append(pi_int_loss.item())
                opt_info.value_loss.append(value_loss.item())
                opt_info.value_int_loss.append(value_int_loss.item())
                opt_info.entropy_loss.append(entropy_loss.item())
                opt_info.entropy_ext_loss.append(entropy_ext_loss.item())
                opt_info.entropy_int_loss.append(entropy_int_loss.item())
                opt_info.utility_nw.append(utility_nw.item())
                opt_info.kl_constraint.append(kl_constraint.item())

                intrinsic = False
                if self.curiosity_type in {'icm', 'micm'}:
                    inv_loss, forward_loss = curiosity_losses
                    opt_info.inv_loss.append(inv_loss.item())
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'disagreement':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'ndigo':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'rnd':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                if intrinsic:
                    opt_info.intrinsic_rewards.append(self.intrinsic_rewards.flatten())
                    opt_info.extint_ratio.append(self.extint_ratio.flatten())

                if self.normalize_extreward:
                    opt_info.normalized_extreward.append(self.normalized_extreward.flatten())
                if self.normalize_intreward:
                    opt_info.normalized_intreward.append(self.normalized_intreward.flatten())
                if self.rescale_extreward:
                    opt_info.rescaled_extreward.append(self.rescaled_extreward.flatten())
                if self.rescale_intreward:
                    opt_info.rescaled_intreward.append(self.rescaled_intreward.flatten())
                opt_info.entropy.append(entropy.item())
                opt_info.entropy_ext.append(entropy_ext.item())
                opt_info.entropy_int.append(entropy_int.item())
                opt_info.perplexity.append(perplexity.item())
                opt_info.perplexity_ext.append(perplexity_ext.item())
                opt_info.perplexity_int.append(perplexity_int.item())
                self.update_counter += 1

        opt_info.return_.append(torch.mean(return_.detach()).detach().clone().item())
        opt_info.return_int_.append(torch.mean(return_int_.detach()).detach().clone().item())
        opt_info.advantage.append(torch.mean(advantage.detach()).detach().clone().item())
        if self.dual_policy_weighting != 'none':
            opt_info.dual_policy_weights.append(dual_policy_weights.clone().data.numpy().flatten())
            opt_info.kl_scores.append(kl_scores.clone().data.numpy().flatten())
            opt_info.advantage_int_weighted.append(self.advantage_int_weighted.flatten())
        if self.dual_value:
            opt_info.advantage_ext.append(torch.mean(advantage_ext.detach()).detach().clone().item())
            opt_info.advantage_int.append(torch.mean(advantage_int.detach()).detach().clone().item())
        opt_info.valpred.append(torch.mean(samples.agent.agent_info.value.detach()).detach().clone().item())

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        layer_info = dict() # empty dict to store model layer weights for tensorboard visualizations

        return opt_info, layer_info

    def optimize_agent(self, itr, samples):
        if self.use_minmax:
          return self._optimize_agent_minmax(itr, samples)
        else:
          return self._optimize_agent(itr, samples)

    def _curiosity_losses(self, agent_curiosity_inputs):
        if self.curiosity_type in {'icm', 'micm'}:
            inv_loss, forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            curiosity_losses = (inv_loss, forward_loss)
        elif self.curiosity_type == 'disagreement':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            curiosity_losses = (forward_loss,)
        elif self.curiosity_type == 'ndigo':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            curiosity_losses = (forward_loss,)
        elif self.curiosity_type == 'rnd':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            curiosity_losses = (forward_loss,)
        else:
            curiosity_losses = []
        return curiosity_losses

    def policy_loss(self, action, new_dist, old_dist, advantage, valid):
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(action, new_dist_info=new_dist, old_dist_info=old_dist) # new_dist_info / old_dist_info
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        return pi_loss

    def value_loss(self, value, return_, valid):
        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        return value_loss

    # def loss_alpha(self,
    #         agent_inputs,
    #         action,
    #         advantage,
    #         valid,
    #         old_dist_ext_info=None,
    #         init_rnn_state=None):

    #     if init_rnn_state is not None:
    #         # [B,N,H] --> [N,B,H] (for cudnn).
    #         init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
    #         init_rnn_state = buffer_method(init_rnn_state, "contiguous")
    #         dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
    #     else:
    #         dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here

    #     dist = self.agent.distribution
    #     pi_loss = self.policy_loss(action, dist_info, old_dist_ext_info, advantage, valid) # \pi(a|s) / pi'(a|s) A(s,a)

    #     return pi_loss


    def loss_min_step(self,
            agent_inputs,
            action,
            return_, # \sum_t r^E_t
            return_int_, # sum_t r^I_t
            advantage, # U^\pi_-(s,a)
            advantage_pi,
            advantage_pi_prime,
            valid,
            old_dist_info=None, # pi(a|s): we need to use old one since it blocks gradients
            init_rnn_state=None):

        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here

        dist = self.agent.distribution
        if self.minmax_ablation == "advantage":
          pi_prime_loss = torch.tensor([0.0])
        elif self.minmax_ablation == "advantage_aux":
          pi_prime_loss = self.policy_loss(action, dist_ext_info, old_dist_info, advantage_pi_prime, valid)
        elif self.minmax_ablation == "decouple": # we only run min_step in decouple mode
          pi_prime_loss = self.policy_loss(action, dist_ext_info, old_dist_info, advantage_pi_prime, valid)
        else:
          pi_prime_loss = self.policy_loss(action, dist_ext_info, old_dist_info, advantage, valid)

        pi_loss = self.policy_loss(action, dist_info, old_dist_info, advantage_pi, valid)
        if self.minmax_ablation == "decouple":
          pi_loss += valid_mean(dist.kl(dist_ext_info, dist_info, detach='first'), valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        value_loss = self.value_loss(value, return_, valid)
        value_int_loss = self.value_loss(value_int, return_int_, valid) if self.dual_value else torch.tensor([0.0])

        entropy = dist.mean_entropy(dist_ext_info, valid)
        perplexity = dist.mean_perplexity(dist_ext_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = (pi_prime_loss + pi_loss) + value_loss + entropy_loss + value_int_loss

        return loss, pi_prime_loss, value_loss, value_int_loss, entropy_loss, entropy, perplexity

    def loss_max_step(self,
            agent_inputs,
            action,
            return_, # \sum_t r^E_t
            advantage, # U^\pi'_+(s,a)
            advantage_pi,
            advantage_pi_prime,
            valid,
            old_dist_ext_info=None, # pi'(a|s): we need to use old one since it blocks gradients
            init_rnn_state=None,
            ret_alpha_loss=False # We use this flag to determine whether to compute alpha loss (for efficiency purppose)
        ):

        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, dist_ext_info, dist_int_info, value, value_int, value_ext_prime = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here

        dist = self.agent.distribution
        if self.minmax_ablation == 'advantage': # To show that performance gain is not from alternating two policies
          pi_loss = torch.tensor([0.0])
        elif self.minmax_ablation == "advantage_aux": # To show that performance gain is not from alternating two policies and optimizing respective objectives
          pi_loss = self.policy_loss(action, dist_info, old_dist_ext_info, advantage_pi, valid)
        else:
          pi_loss = self.policy_loss(action, dist_info, old_dist_ext_info, advantage, valid)
        pi_prime_loss = self.policy_loss(action, dist_ext_info, old_dist_ext_info, advantage_pi_prime, valid)

        with torch.no_grad():
          if ret_alpha_loss:
              alpha_loss = self.policy_loss(action, dist_info, old_dist_ext_info, advantage_pi_prime, valid)
          else:
              alpha_loss = 0

        value_loss = self.value_loss(value_ext_prime, return_, valid)

        entropy = dist.mean_entropy(dist_ext_info, valid)
        perplexity = dist.mean_perplexity(dist_ext_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = (pi_loss + pi_prime_loss) + value_loss + entropy_loss

        return loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, alpha_loss

    def loss(self, agent_inputs, agent_curiosity_inputs, action, return_, return_int_, advantage, advantage_ext, advantage_int, valid, old_dist_info,
            old_dist_ext_info=None, old_dist_int_info=None, init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, dist_ext_info, dist_int_info, value, value_int, _, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, dist_ext_info, dist_int_info, value, value_int, _ = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here

        # combined policy
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        perplexity = dist.mean_perplexity(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        if self.dual_policy != 'default':
            # extrinsic policy
            dist_ext = self.agent.distribution_ext
            ratio_ext = dist_ext.likelihood_ratio(action, old_dist_info=old_dist_ext_info, new_dist_info=dist_ext_info)
            surr_1_ext = ratio_ext * advantage_ext
            clipped_ratio_ext = torch.clamp(ratio_ext, 1. - self.ratio_clip, 1. + self.ratio_clip)
            surr_2_ext = clipped_ratio_ext * advantage_ext
            surrogate_ext = torch.min(surr_1_ext, surr_2_ext)
            if self.importance_sample > 0.:
                ext_importance = dist_ext.likelihood_ratio(action, old_dist_info=dist_info, new_dist_info=dist_ext_info, stopgrad='both')
                ext_importance = torch.clamp(ext_importance, 1. - self.importance_sample, 1. + self.importance_sample).clone().detach()
                surrogate_ext *= ext_importance
            pi_ext_loss = - valid_mean(surrogate_ext, valid)

            entropy_ext = dist_ext.mean_entropy(dist_ext_info, valid)
            perplexity_ext = dist_ext.mean_perplexity(dist_ext_info, valid)
            entropy_ext_loss = - self.entropy_loss_coeff * entropy_ext

            # intrinsic policy
            if self.dual_policy_noint:
                pi_int_loss = torch.tensor([0.0])
                entropy_int = torch.tensor([0.0])
                perplexity_int = torch.tensor([0.0])
                entropy_int_loss = torch.tensor([0.0])
            else:
                dist_int = self.agent.distribution_int
                ratio_int = dist_int.likelihood_ratio(action, old_dist_info=old_dist_int_info, new_dist_info=dist_int_info)
                surr_1_int = ratio_int * advantage_int
                clipped_ratio_int = torch.clamp(ratio_int, 1. - self.ratio_clip, 1. + self.ratio_clip)
                surr_2_int = clipped_ratio_int * advantage_int
                surrogate_int = torch.min(surr_1_int, surr_2_int)
                pi_int_loss = - valid_mean(surrogate_int, valid)

                entropy_int = dist_int.mean_entropy(dist_int_info, valid)
                perplexity_int = dist_int.mean_perplexity(dist_int_info, valid)
                entropy_int_loss = - self.entropy_loss_coeff * entropy_int

        if self.dual_value:
            value_int_error = 0.5 * (value_int - return_int_) ** 2
            value_int_loss = self.value_loss_coeff * valid_mean(value_int_error, valid)
            loss = pi_loss + value_loss + value_int_loss + entropy_loss
            if self.dual_policy in {'combined', 'ext', 'int'}:
                loss += pi_ext_loss + entropy_ext_loss
                if self.dual_policy_noint == False:
                    loss += pi_int_loss + entropy_int_loss
        else:
            value_int_loss = torch.tensor([0.0])
            loss = pi_loss + value_loss + entropy_loss

        if self.utility_noworkers:
            explore_ratio = dist.likelihood_ratio(action, old_dist_info=dist_ext_info, new_dist_info=dist_info, stopgrad=self.util_detach)
            clipped_explore_ratio = torch.clamp(explore_ratio, min=1.0-self.util_clamp, max=1.0+self.util_clamp)
            utility_nw = clipped_explore_ratio * advantage_ext
            utility_nw = valid_mean(utility_nw, valid)
            if self.kl_clamp > 0.0:
                kl_constraint = torch.clamp(dist.kl(dist_info, dist_ext_info, detach=self.kl_detach), min=0.0, max=self.kl_clamp)
            else:
                kl_constraint = dist.kl(dist_info, dist_ext_info, detach=self.kl_detach)
            kl_constraint = - self.kl_lambda * valid_mean(kl_constraint, valid)
            loss += utility_nw
            loss += kl_constraint
        else:
            utility_nw = torch.tensor([0.0])
            kl_constraint = torch.tensor([0.0])

        if self.curiosity_type in {'icm', 'micm'}:
            inv_loss, forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += inv_loss
            loss += forward_loss
            curiosity_losses = (inv_loss, forward_loss)
        elif self.curiosity_type == 'disagreement':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        elif self.curiosity_type == 'ndigo':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        elif self.curiosity_type == 'rnd':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        else:
            curiosity_losses = None

        if self.dual_value and self.dual_policy in {'combined', 'ext', 'int'}:
            pass
        elif self.dual_value:
            pi_ext_loss = torch.tensor([0.0])
            pi_int_loss = torch.tensor([0.0])
            entropy_ext_loss = torch.tensor([0.0])
            entropy_ext = torch.tensor([0.0])
            perplexity_ext = torch.tensor([0.0])
            entropy_int_loss = torch.tensor([0.0])
            entropy_int = torch.tensor([0.0])
            perplexity_int = torch.tensor([0.0])
        else:
            pi_ext_loss = torch.tensor([0.0])
            pi_int_loss = torch.tensor([0.0])
            entropy_ext_loss = torch.tensor([0.0])
            entropy_ext = torch.tensor([0.0])
            perplexity_ext = torch.tensor([0.0])
            entropy_int_loss = torch.tensor([0.0])
            entropy_int = torch.tensor([0.0])
            perplexity_int = torch.tensor([0.0])
            value_int_loss = torch.tensor([0.0])

        return loss, pi_loss, pi_ext_loss, pi_int_loss, value_loss, value_int_loss, entropy_loss, entropy_ext_loss, entropy_int_loss, entropy, entropy_ext, entropy_int, perplexity, perplexity_ext, perplexity_int, curiosity_losses, utility_nw, kl_constraint




