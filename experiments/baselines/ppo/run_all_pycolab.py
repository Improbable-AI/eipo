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
  runs = [1, 2, 3]

  common_exp_args = [
    "-alg ppo",
    "-curiosity_alg none",
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
    "-launch_tmux no",
    "-use_wandb",
    "-model_save_freq 0"
  ]

  all_job_args = []
  for job_idx, (n_tasks, env, run) in enumerate(sweep(itertools.product(envs, runs),
                                                      n_parallel_task=launch_args.n_parallel_task)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [] + common_exp_args
      args.append(f"-env {env[task_idx]}")
      args.append(f"-log_dir ./results/normalize_reward/ppo/{env[task_idx]}/orig/ppo_{env[task_idx]}/{exp_id()}/run_{run[task_idx]}")
      args.append(f"-gpu_id {launch_args.gpus[job_idx % len(launch_args.gpus)]}")
      job_args.append(" ".join(args))
    all_job_args.append(job_args[0])

  print(f"Total: {len(all_job_args)}")

  if launch_args.task_id == ":":
    start_task_id = 0
    end_task_id = len(all_job_args) - 1
  else:
    start_task_id, end_task_id = map(lambda x: int(x), launch_args.task_id.split(":"))

  launch("toy_ppo", all_job_args[start_task_id: end_task_id + 1],
      mode=launch_args.mode,
      parallel=False,
      verbose=True)
