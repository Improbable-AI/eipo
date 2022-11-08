import os
import sys
import copy
import itertools
import datetime
from pathlib import Path

import gym
import uuid
import multiprocessing
import glob

import atari_py
from experiments import exp_id, launch, sweep, get_launch_args

ROM_PATH = atari_py.__path__[0] + "/atari_roms"

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}"
  launch_args = get_launch_args(experiment)

  # Hyperparameters
  # NOTE: we load env list from txt to ensure order is the same at every machine. we don't sort because we've launched a lot of jobs using supercloud order.
  # envs = [(os.path.basename(rom).split(".")[0]) + "\n" for rom in glob.glob(ROM_PATH + "/*")]
  # with open("atari_envs.txt", "w") as f:
  #     f.writelines(envs)

  # with open("atari_envs.txt", "r") as f:
  #   envs = [env.rstrip() for env in f]

  # envs = ['atlantis', "adventure", "defender"]
  envs = ['montezuma_revenge', 'venture',]
  minmax_alphas = [0.5,]
  alpha_lrs = [0.005,]
  runs = [1, 2, 3,]

  common_exp_args = [
    "-alg ppo",
    "-curiosity_alg rnd",
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
    "-drop_probability 0.25",
    "-max_episode_steps 27000",
    "-launch_tmux no",
    "-score_multiplier 1",
    "-normalize_obs",
    "-normalize_advantage",
    "-normalize_extreward", # We are testing this
    "-prediction_beta 1",
    "-dual_value",
    "-use_wandb"
  ]

  # Create sweep
  all_job_args = []
  for job_idx, (n_tasks, env, minmax_alpha, alpha_lr, run) in enumerate(
                                                      sweep(itertools.product(envs, minmax_alphas, alpha_lrs, runs),
                                                      n_parallel_task=1)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [] + common_exp_args
      args.append(f"-gpu_id {launch_args.gpus[job_idx % len(launch_args.gpus)]}")
      args.append(f"-env {env[task_idx]}")
      args.append("-iterations 49152000" if env[task_idx] != "montezuma_revenge" else "-iterations 98304000")
      args.append(f"-minmax_alpha {minmax_alpha[task_idx]}")
      args.append(f"-alpha_lr {alpha_lr[task_idx]}")
      args.append(f"-log_dir ./results/normalize_ext/ppo/{env[task_idx]}/winlose_adapt_minmax/ppo_minmax_rnd_{minmax_alpha[task_idx]}_{alpha_lr[task_idx]}_{env[task_idx]}/{exp_id()}/run_{run[task_idx]}")
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
