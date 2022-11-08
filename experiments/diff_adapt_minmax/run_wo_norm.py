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
    # 'video_pinball',
    # 'pooyan',
    # "skiing",
    # 'space_invaders',
    # 'up_n_down'
    # 'ms_pacman',
    # "kung_fu_master",
    # 'road_runner',
    # 'battle_zone'
    # 'double_dunk',
    # 'boxing',
    # "venture",
    # "star_gunner",
    # "yars_revenge",
    "montezuma_revenge",
    # "jamesbond",
    # "asterix",
    # "zaxxon",
    # "enduro",
    # "frostbite",
    # "asterix",

    # 2nd set
    # 'robotank',
    # "tutankham",
    # "krull",
    # "gravitar",
    # "kangaroo",
    # "kaboom",
    # "bowling",
    # "amidar",

    # 3rd set
    # "solaris",
    # "hero",
    # "qbert",
    # "seaquest",
    # "enduro",
    # "pitfall",
    # "breakout",

    # 4th set
    # "adventure",
    # "air_raid",
    # "alien",
    # "assault",
    # "asteroids",
    # "atlantis",
    # "chopper_command",
    # "gopher",
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
    # "-normalize_extreward", # We are testing this
    "-prediction_beta 1",
    "-dual_value",
    "-use_wandb",
    "-model_save_freq 0",
  ]

  for job_idx, (n_tasks, env, minmax_alpha, alpha_lr, run) in enumerate(sweep(itertools.product(envs, minmax_alphas, alpha_lrs, runs),
                                                      n_parallel_task=launch_args.n_parallel_task)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [] + common_exp_args
      args.append(f"-env {env[task_idx]}")
      args.append("-iterations 49152000" if env[task_idx] != "montezuma_revenge" else "-iterations 98304000")
      args.append(f"-minmax_alpha {minmax_alpha[task_idx]}")
      args.append(f"-alpha_lr {alpha_lr[task_idx]}")
      args.append(f"-log_dir ./results/normalize_ext/ppo/{env[task_idx]}/diff_adapt_minmax/ppo_minmax_rnd_{minmax_alpha[task_idx]}_{alpha_lr[task_idx]}_{env[task_idx]}/{exp_id()}/run_{run[task_idx]}")
      args.append(f"-gpu_id {launch_args.gpus[job_idx % len(launch_args.gpus)]}")
      job_args.append(" ".join(args))

    launch(experiment, job_args,
      mode=launch_args.mode,
      verbose=True)

  print(f"Launched {job_idx + 1} jobs")