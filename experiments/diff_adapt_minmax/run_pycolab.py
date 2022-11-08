import os
import sys
import copy
import itertools
import datetime
from pathlib import Path

import gym
from experiments import get_launch_args, sweep, launch, exp_id

import multiprocessing

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}"
  launch_args = get_launch_args(experiment)

  # Hyperparameters
  envs = [
   "Misaligned-v0",
  ]
  minmax_alphas = [0.5,]
  alpha_lrs = [0.005,]
  # minmax_alphas = [0.0001,]
  runs = [1,]

  common_exp_args = [
    "-alg ppo",
    "-curiosity_alg rnd",
    "-use_minmax",
    "-use_adapt_alpha",
    "-alpha_clip=g-0.05,0.05",
    "-minmax_switch=diff",
    "-iterations 25000000",
    "-lstm",
    "-num_envs 64",
    "-sample_mode gpu",
    "-num_gpus 1",
    "-num_cpus {}".format(multiprocessing.cpu_count()),
    "-eval_envs 0",
    "-eval_max_steps 51000",
    "-eval_max_traj 50",
    "-timestep_limit 500",
    "-log_interval 10000",
    "-record_freq 0",
    "-pretrain None",
    "-discount 0.99",
    "-lr 0.0001",
    "-v_loss_coeff 1.0",
    "-entropy_loss_coeff 0.001",
    "-grad_norm_bound 1.0",
    "-gae_lambda 0.95",
    "-minibatches 4",
    "-epochs 3",
    "-ratio_clip 0.1",
    "-normalize_advantage",
    "-normalize_reward",
    # "-normalize_extreward",
    "-dual_policy default",
    "-dual_policy_weighting none",
    # "-log_heatmaps",
    "-normalize_obs",
    "-obs_type rgb",
    "-max_episode_steps 500",
    "-feature_encoding none",
    "-prediction_beta 1000",
    "-drop_probability 0.5",
    "-dual_value",
    "-discount_ri 0.99",
    "-launch_tmux no",
    "-use_wandb"
  ]

  for job_idx, (n_tasks, env, minmax_alpha, alpha_lr, run) in enumerate(sweep(itertools.product(envs, minmax_alphas, alpha_lrs, runs),
                                                      n_parallel_task=launch_args.n_parallel_task)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [] + common_exp_args
      args.append(f"-env {env[task_idx]}")
      args.append(f"-minmax_alpha {minmax_alpha[task_idx]}")
      args.append(f"-alpha_lr {alpha_lr[task_idx]}")
      args.append(f"-log_dir ./results/normalize_reward/ppo/{env[task_idx]}/diff_adapt_minmax/ppo_minmax_rnd_{minmax_alpha[task_idx]}_{alpha_lr[task_idx]}_{env[task_idx]}/{exp_id()}/run_{run[task_idx]}")
      args.append(f"-gpu_id {launch_args.gpus[job_idx % len(launch_args.gpus)]}")
      job_args.append(" ".join(args))

    launch(experiment, job_args,
      mode=launch_args.mode,
      verbose=True)

  print(f"Launched {job_idx + 1} jobs")