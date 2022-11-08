import os
import sys
import copy
import itertools
import datetime
from pathlib import Path
import multiprocessing

from experiments import get_launch_args, sweep, launch

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}"
  launch_args = get_launch_args(experiment)

  # Hyperparameters
  envs = [
    # "boxing",
    # "venture",
    "star_gunner",
    # "jamesbond",
    # "asterix",
    # "zaxxon",
    # "hero",
    # "qbert",
  ]
  runs = [ 3]

  common_exp_args = [
    "-alg ppo",
    "-curiosity_alg icm",

    "-lr 0.0001",
    "-entropy_loss_coeff 0.001",
    "-minibatches 4",
    "-feature_encoding none",
    "-lstm",
    "-num_envs 128",
    "-sample_mode gpu",
    "-num_gpus 1",
    # "-num_cpus {}".format(multiprocessing.cpu_count()),
    "-num_cpus 20",
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

  for job_idx, (n_tasks, env, run) in enumerate(sweep(itertools.product(envs, runs),
                                                      n_parallel_task=launch_args.n_parallel_task)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [] + common_exp_args
      args.append(f"-env {env[task_idx]}")
      args.append(f"-log_dir ./results/normalize_ext/ppo/{env[task_idx]}/rnd/ppo_icm_{env[task_idx]}/run_{run[task_idx]}")
      args.append(f"-gpu_id {launch_args.gpus[job_idx % len(launch_args.gpus)]}")
      job_args.append(" ".join(args))

    launch(experiment, job_args,
      mode=launch_args.mode,
      verbose=True)

  print(f"Launched {job_idx + 1} jobs")