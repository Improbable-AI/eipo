import copy
from datetime import datetime
import jaynes

from rlpyt.utils.launching.launcher import start_experiment

from rlpyt.utils.launching.arguments import get_base_args

def get_rnd_ppo_atari_base_args():
  base_args = get_base_args(alg="ppo", env="breakout", curiosity_alg="rnd")
  base_args.alg = "ppo"
  base_args.curiosity_alg = "rnd"
  base_args.lr = 0.0001
  base_args.entropy_loss_coeff = 0.001
  base_args.minibatches = 4
  base_args.feature_encoding = "none"
  base_args.iterations = 49152000
  base_args.lstm = True
  base_args.num_envs = 128
  base_args.sample_mode = "gpu"
  base_args.gpu_id = 'auto'
  base_args.num_gpus = 1
  base_args.num_cpus = 21
  base_args.eval_envs = 0
  base_args.eval_max_steps = 51000
  base_args.eval_max_traj = 50
  base_args.timestep_limit = 128
  base_args.log_interval = 10000
  base_args.record_freq = 5
  base_args.pretrain = "None"
  base_args.discount = 0.99
  base_args.v_loss_coeff = 1.0
  base_args.grad_norm_bound = 1.0
  base_args.gae_lambda = 0.95
  base_args.epochs = 4
  base_args.ratio_clip = 0.1
  base_args.drop_probability = 0.25
  base_args.max_episode_steps = 27000
  base_args.score_multiplier = 1
  base_args.normalize_obs = True
  base_args.normalize_advantage = True
  base_args.prediction_beta = 1
  base_args.dual_value = True
  base_args.launch_tmux = False
  base_args.use_wandb = True

  return base_args

def timestamp():
  now = datetime.now() # current date and time
  date_time = now.strftime("%Y/%m/%d/%H-%M-%S/")
  return date_time


if __name__ == "__main__":
    '''
    Modify here to your own directory
    '''
    RESULT_DIR = "/data/pulkitag/misc/zwhong" # For vision gpu
    # RESULT_DIR = "/home/gridsan/zwhong" # For supercloud

    # NOTE: breakout is a dummy env for getting the base args for Atari
    base_args = get_rnd_ppo_atari_base_args()

    # Sweep over environments
    for run_id in range(1):
      for env in ["venture",]:
        args = copy.deepcopy(base_args)
        args.env = env
        args.result_dir = args.log_dir = f"{RESULT_DIR}/results/{timestamp()}/ppo_{env}/{run_id}" # TODO: set it to NFS

        jaynes.config(
          entry_script="mpirun python -u -m jaynes.entry",
          verbose=False)
        jaynes.run(start_experiment, args, use_jaynes=True)

      jaynes.listen()
