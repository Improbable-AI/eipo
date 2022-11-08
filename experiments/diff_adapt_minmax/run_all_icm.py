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
    "kaboom",
    "star_gunner",
    "time_pilot",
    "enduro",
    "yars_revenge",
    "jamesbond"
  ]
  minmax_alphas = [0.5,]
  alpha_lrs = [0.005,]
  # minmax_alphas = [0.0001,]
  runs = [1, 2, 3, 4, 5]

  common_exp_args = [
    "-alg ppo",
    "-curiosity_alg icm",

    "-use_minmax",
    "-use_adapt_alpha",
    "-alpha_clip=g-0.05,0.05",
    "-minmax_switch=diff",

    "-lr 0.0001",
    "-entropy_loss_coeff 0.001",
    "-minibatches 4",
    "-feature_encoding none",
    "-lstm",
    "-num_envs 128",
    "-sample_mode gpu",
    "-num_gpus 1",
    "-num_cpus {}".format(multiprocessing.cpu_count()),
    # "-num_cpus 20",
    "-eval_envs 0",
    "-eval_max_steps 51000",
    "-eval_max_traj 50",
    "-timestep_limit 128",
    "-log_interval 10000",
    "-record_freq 0",
    "-pretrain None",
    "-discount 0.99",
    "-v_loss_coeff 1.0",
    "-grad_norm_bound 1.0",
    "-gae_lambda 0.95",
    "-epochs 4",
    "-ratio_clip 0.1",
    "-max_episode_steps 27000",
    "-launch_tmux no",
    "-score_multiplier 1",
    "-normalize_advantage",
    "-normalize_reward", # We are testing this
    "-normalize_extreward", # We are testing this
    "-normalize_intreward", # We are testing this
    "-prediction_beta 1",
    "-feature_encoding idf_burda",
    "-forward_loss_wt 0.2",
    "-batch_norm",
    "-forward_model res",
    "-feature_space inverse",
    "-encoder_pretrain None",
    "-use_wandb",
    "-model_save_freq 0",
  ]

  all_job_args = []
  for job_idx, (n_tasks, env, minmax_alpha, alpha_lr, run) in enumerate(sweep(itertools.product(envs, minmax_alphas, alpha_lrs, runs),
                                                      n_parallel_task=launch_args.n_parallel_task)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [] + common_exp_args
      args.append(f"-env {env[task_idx]}")
      args.append("-iterations 49152000" if env[task_idx] != "montezuma_revenge" else "-iterations 98304000")
      args.append(f"-minmax_alpha {minmax_alpha[task_idx]}")
      args.append(f"-alpha_lr {alpha_lr[task_idx]}")
      args.append(f"-log_dir ./results/normalize_ext/ppo/{env[task_idx]}/diff_adapt_minmax/ppo_minmax_icm_{minmax_alpha[task_idx]}_{alpha_lr[task_idx]}_{env[task_idx]}/{exp_id()}/run_{run[task_idx]}")
      args.append(f"-gpu_id {launch_args.gpus[job_idx % len(launch_args.gpus)]}")
      job_args.append(" ".join(args))
    all_job_args.append(job_args[0])

  print(f"Total: {len(all_job_args)}")

  if launch_args.task_id == ":":
    start_task_id = 0
    end_task_id = len(all_job_args) - 1
  else:
    start_task_id, end_task_id = map(lambda x: int(x), launch_args.task_id.split(":"))

  launch("diff_adapt_minmax", all_job_args[start_task_id: end_task_id + 1],
      mode=launch_args.mode,
      parallel=False,
      verbose=True)