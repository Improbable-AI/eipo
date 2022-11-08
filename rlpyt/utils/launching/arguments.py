import sys
import argparse
import json

with open('./global.json') as global_params:
    params = json.load(global_params)
    _ATARI_ENVS = params['envs']['atari_envs']
    _PYCOLAB_ENVS = params['envs']['pycolab_envs']
    _MUJOCO_ENVS = params['envs']['mujoco_envs']

def get_args(args_in=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument('-alg', type=str, choices=['ppo', 'sac', 'a2c'], help='Which learning algorithm to run.')
    parser.add_argument('-curiosity_alg', type=str, choices=['none', 'icm', 'micm', 'disagreement', 'ndigo', 'rnd'], help='Which intrinsic reward algorithm to use.')
    parser.add_argument('-env', type=str, help='Which environment to run on.')

    # general args
    parser.add_argument('-iterations', default=int(1e8), type=int, help='Number of optimization iterations to run (global timesteps).')
    parser.add_argument('-lstm', action='store_true', help='Whether or not to run an LSTM or FF policy.')
    parser.add_argument('-no_extrinsic', action='store_true', help='Whether or not to use no extrinsic reward.')
    parser.add_argument('-no_negative_reward', action='store_true', help='Whether or not to use negative rewards (living penalty for example).')
    parser.add_argument('-num_envs', default=4, type=int, help='Number of environments to run in parallel.')
    parser.add_argument('-sample_mode', default='cpu', type=str, help='Whether to use GPU or CPU sampling.')
    parser.add_argument('-gpu_id', default=0, type=int, help='GPU ID. If auto, will use the first available GPU.')
    parser.add_argument('-num_gpus', default=0, type=int, help='Number of GPUs available.')
    parser.add_argument('-num_cpus', default=1, type=int, help='Number of CPUs to run worker processes.')
    parser.add_argument('-eval_envs', default=0, type=int, help='Number of evaluation environments per worker process.')
    parser.add_argument('-eval_max_steps', default=int(51e3), type=int, help='Max number of timesteps run during an evaluation cycle (from one evaluation process).')
    parser.add_argument('-eval_max_traj', default=50, type=int, help='Max number of trajectories collected during an evaluation cycle (from all evaluation processes).')
    parser.add_argument('-timestep_limit', default=20, type=int, help='Max number of timesteps per trajectory')

    # logging args
    parser.add_argument('-log_interval', default=int(1e4), type=int, help='Number of environment steps between logging events.')
    parser.add_argument('-model_save_freq', default=1, type=int, help='How often to save model weights (in batches)')
    parser.add_argument('-record_freq', default=0, type=int, help='Interval between video recorded episodes (in episodes). 0 means dont record.')
    parser.add_argument('-pretrain', default="None", help='The directory to draw model parameters from if restarting an experiment. If None start a new experiment.')
    parser.add_argument('-log_dir', default=None, type=str, help='Directory where videos/models/etc are logged. If none, this will be generated at launch time.')
    parser.add_argument('-result_dir', default=None, type=str, help='Directory where the results are stored. If None, use global.json[local_result_dir]')
    parser.add_argument('-use_wandb', action='store_true', default=False, help='Whether use W&B logger')

    # learning algorithm specific args
    if 'ppo' in args_in:
        parser.add_argument('-discount', default=0.99, type=float, help='Reward discount factor applied.')
        parser.add_argument('-lr', default=0.0001, type=float, help='Learning rate.')
        parser.add_argument('-v_loss_coeff', default=1.0, type=float, help='Value function coefficient in the loss function.')
        parser.add_argument('-entropy_loss_coeff', default=0.001, type=float, help='Entropy coefficient in the loss function.')
        parser.add_argument('-grad_norm_bound', default=1.0, type=float, help='Gradient norm clipping bound.')
        parser.add_argument('-gae_lambda', default=0.95, type=float, help='Bias/variance tradeoff for GAE.')
        parser.add_argument('-minibatches', default=1, type=int, help='Number of minibatches per iteration.')
        parser.add_argument('-epochs', default=3, type=int, help='Number of passes over minibatches per iteration.')
        parser.add_argument('-ratio_clip', default=0.1, type=float, help='The policy ratio (new vs old) clipping bound.')
        parser.add_argument('-linear_lr', action='store_true', help='Whether or not to change learning rate linearly as a function of iteration.')
        parser.add_argument('-normalize_advantage', action='store_true', help='Whether or not to normalize advantages.')
        parser.add_argument('-normalize_reward', action='store_true', help='Whether or not to normalize total rewards before computing advantages')
        parser.add_argument('-normalize_extreward', action='store_true', help='Whether or not to normalize extrinsic rewards before computing advantages')
        parser.add_argument('-normalize_intreward', action='store_true', help='Whether or not to normalize intrinsic rewards before computing advantages')
        parser.add_argument('-rescale_extreward', action='store_true', help='Whether or not to rescale extrinsic rewards to 0-1 before computing advantages')
        parser.add_argument('-rescale_intreward', action='store_true', help='Whether or not to rescale intrinsic rewards to 0-1 before computing advantages')
        parser.add_argument('-dual_policy', default='default', type=str, choices=['default', 'combined', 'int', 'ext'], help='What policy to sample - default is just a single policy, trained with re+ri')
        parser.add_argument('-dual_policy_noint', action='store_true', help='Whether or not to not train the intrinsic policy')
        parser.add_argument('-dual_policy_weighting', default='none', type=str, choices=['none', 'ext_first', 'int_first'], help='Which order to compute KL if weighting intrinsic rewards using the dual policy method')
        parser.add_argument('-dpw_formulation', default='inverse', type=str, choices=['inverse', 'exp'], help='Which formulation to use when computing weights from KL scores')
        parser.add_argument('-utility_noworkers', action='store_true', help='Whether or not to use intrinsic utility in the loss function')
        parser.add_argument('-kl_lambda', default=1.0, type=float, help='Scaling factor for kl constraint when using utility')
        parser.add_argument('-kl_clamp', default=0.0, type=float, help='Whether or not to clamp kl constraint')
        parser.add_argument('-util_clamp', default=0.1, type=float, help='How much to clamp the utility term around 1.0')
        parser.add_argument('-util_detach', default='none', type=str, help='Whether or not to stop backprop in numerator or denominator of utility term')
        parser.add_argument('-kl_detach', default='none', type=str, choices=['none', 'first', 'second'], help='Detach one of the policies kl constraint term')
        parser.add_argument('-importance_sample', default=0.0, type=float, help='Whether or not to use importance sampling to train the extrinsic policy when using utility')
        # For minmax
        parser.add_argument('-use_minmax', default=False, action='store_true', help='Whether to use minmax formulation.')
        parser.add_argument('-minmax_alpha', default='0.1', type=str, help='Largrangian multiplier for minmax approach')
        parser.add_argument('-use_adapt_alpha', default=False, action='store_true', help='Whether to use adaptive alpha in minmax formulation.')
        parser.add_argument('-alpha_lr', default=0.001, type=float, help='Largrangian multiplier learning rate')
        parser.add_argument('-minmax_ablation', default='none', choices=['none', 'advantage', 'advantage_aux', 'decouple'])
        parser.add_argument('-alpha_clip', default='none', help="none if not used, otherwise format is {lb},{ub}")
        parser.add_argument('-minmax_switch', default='none', help='The policy alternating scheme', choices=['none', 'win-lose', 'diff'])
    elif 'a2c' in args_in:
        parser.add_argument('-discount', default=0.99, type=float, help='Reward discount factor applied.')
        parser.add_argument('-lr', default=0.001, type=float, help='Learning rate.')
        parser.add_argument('-v_loss_coeff', default=0.5, type=float, help='Value function coefficient in the loss function.')
        parser.add_argument('-entropy_loss_coeff', default=0.01, type=float, help='Entropy coefficient in the loss function.')
        parser.add_argument('-grad_norm_bound', default=1.0, type=float, help='Gradient norm clipping bound.')
        parser.add_argument('-gae_lambda', default=1.0, type=float, help='Bias/variance tradeoff for GAE.')
        parser.add_argument('-normalize_advantage', action='store_true', help='Whether or not to normalize advantages.')

    # environment specific args
    environment = args_in[args_in.index('-env')+1]
    if 'mario' in environment.lower():
        parser.add_argument('-mario_level', default='Level1-1', type=str, help='World and level to start at for super mario bros.')
        parser.add_argument('-normalize_obs', action='store_true', help='Whether or not to normalize the observation each step.')
    elif environment in _PYCOLAB_ENVS:
        parser.add_argument('-log_heatmaps', action='store_true', help='Whether or not to store heatmaps.')
        parser.add_argument('-normalize_obs', action='store_true', help='Whether or not to normalize the observation each step.')
        parser.add_argument('-obs_type', default='mask', type=str, choices=['mask', 'rgb', 'rgb_full'], help='Whether to pass binary mask observations or RGB observations.')
        parser.add_argument('-grayscale', action='store_true', help='Whether or not to grayscale images if using rgb.')
        parser.add_argument('-max_episode_steps', default=500, type=int, help='How many steps to run before the done flag is raised.')
    elif environment in _ATARI_ENVS:
        parser.add_argument('-max_episode_steps', default=27000, type=int, help='How many steps to run before the done flag is raised.')
        parser.add_argument('-normalize_obs', action='store_true', help='Whether or not to normalize the observation each step.')
        parser.add_argument('-score_multiplier', default=1.0, type=float, help='A multiplier for the extrinsic reward.')
        parser.add_argument('-repeat_action_probability', default=0.0, type=float, help='Probability that an action will repeat (sticky actions).')
        parser.add_argument('-fire_on_reset', action='store_true', help='Whether or not to automatically press the fire button to start the game, or have the agent do this.')

    # curiosity specific args
    curiosity_alg = args_in[args_in.index('-curiosity_alg')+1]
    if curiosity_alg == 'icm':
        parser.add_argument('-feature_encoding', default='idf_burda', type=str, choices=['none', 'idf', 'idf_burda', 'idf_maze'], help='Which feature encoding method to use with ICM.')
        parser.add_argument('-forward_loss_wt', default=0.2, type=float, help='Forward loss coefficient. Inverse weight is (1 - this).')
        parser.add_argument('-batch_norm', action='store_true', help='Whether or not to use batch norm in the feature encoder.')
        parser.add_argument('-prediction_beta', default=1.0, type=float, help='Scalar multiplier applied to the prediction error to generate the intrinsic reward. Environment dependent.')
        parser.add_argument('-forward_model', default='res', type=str, choices=['res', 'og'], help='Which forward model architecture to use.')
        parser.add_argument('-feature_space', default='inverse', type=str, choices=['inverse', 'random'], help='Use inverse features or random fixed features.')
        parser.add_argument('-encoder_pretrain', default=None, help='The path to load an encoder model, and None if training from scratch.')
        parser.add_argument('-fix_features', action='store_true', help='Whether or not to fix the feature space during training.')
        parser.add_argument('-dual_value', action='store_true', help='Whether or not to train separate value functions for extrinsic and intrinsic rewards.')
        parser.add_argument('-discount_ri', default=0.99, type=float, help='Sets intrinsic reward discounting. If dual value is not being used, make this equal to the extrinsic discount.')
    elif curiosity_alg == 'micm':
        parser.add_argument('-feature_encoding', default='idf_burda', type=str, choices=['none', 'idf', 'idf_burda', 'idf_maze'], help='Which feature encoding method to use with ICM.')
        parser.add_argument('-forward_loss_wt', default=0.2, type=float, help='Forward loss coefficient. Inverse weight is (1 - this).')
        parser.add_argument('-batch_norm', action='store_true', help='Whether or not to use batch norm in the feature encoder.')
        parser.add_argument('-prediction_beta', default=1.0, type=float, help='Scalar multiplier applied to the prediction error to generate the intrinsic reward. Environment dependent.')
        parser.add_argument('-forward_model', default='res', type=str, choices=['res', 'og'], help='Which forward model architecture to use.')
        parser.add_argument('-ensemble_mode', default='sample', type=str, choices=['sample', 'mean', 'var'], help='Which ensemble reward formulation to use.')
        parser.add_argument('-dual_value', action='store_true', help='Whether or not to train separate value functions for extrinsic and intrinsic rewards.')
        parser.add_argument('-discount_ri', default=0.99, type=float, help='Sets intrinsic reward discounting. If dual value is not being used, make this equal to the extrinsic discount.')
    elif curiosity_alg == 'disagreement':
        parser.add_argument('-feature_encoding', default='idf_burda', type=str, choices=['none', 'idf', 'idf_burda', 'idf_maze'], help='Which feature encoding method to use with ICM.')
        parser.add_argument('-forward_loss_wt', default=0.2, type=float, help='Forward loss coefficient. Inverse weight is (1 - this).')
        parser.add_argument('-ensemble_size', default=5, type=int, help='Number of forward models used to compute disagreement reward.')
        parser.add_argument('-batch_norm', action='store_true', help='Whether or not to use batch norm in the feature encoder.')
        parser.add_argument('-prediction_beta', default=1.0, type=float, help='Scalar multiplier applied to the prediction error to generate the intrinsic reward. Environment dependent.')
        parser.add_argument('-forward_model', default='res', type=str, choices=['res', 'og'], help='Which forward model architecture to use.')
        parser.add_argument('-dual_value', action='store_true', help='Whether or not to train separate value functions for extrinsic and intrinsic rewards.')
        parser.add_argument('-discount_ri', default=0.99, type=float, help='Sets intrinsic reward discounting. If dual value is not being used, make this equal to the extrinsic discount.')
    elif curiosity_alg == 'ndigo':
        parser.add_argument('-feature_encoding', default='idf_maze', type=str, choices=['none', 'idf', 'idf_burda', 'idf_maze'], help='Which feature encoding method to use with ICM.')
        parser.add_argument('-pred_horizon', default=1, type=int, help='Number of prediction steps used to calculate intrinsic reward.')
        parser.add_argument('-batch_norm', action='store_true', help='Whether or not to use batch norm in the feature encoder.')
        parser.add_argument('-prediction_beta', default=1.0, type=float, help='Scalar multiplier applied to the prediction error to generate the intrinsic reward. Environment dependent.')
    elif curiosity_alg == 'rnd':
        parser.add_argument('-feature_encoding', default='none', type=str, choices=['none'], help='Which feature encoding method to use with RND.')
        parser.add_argument('-prediction_beta', default=1.0, type=float, help='Scalar multiplier applied to the prediction error to generate the intrinsic reward. Environment dependent.')
        parser.add_argument('-drop_probability', default=1.0, type=float, help='Decimal percent of experience to drop when training the predictor model.')
        parser.add_argument('-dual_value', action='store_true', help='Whether or not to train separate value functions for extrinsic and intrinsic rewards.')
        parser.add_argument('-discount_ri', default=0.99, type=float, help='Sets intrinsic reward discounting. If dual value is not being used, make this equal to the extrinsic discount.')
    elif curiosity_alg == 'none':
        parser.add_argument('-feature_encoding', default='none', type=str, choices=['none'], help='Which feature encoding method to use with your policy.')


    # switch argument (only used in launch.py in __main__)
    parser.add_argument('-launch_tmux', default='yes', type=str, help='')

    return parser.parse_args(args=args_in)

def get_base_args(alg: str, env: str, curiosity_alg: str):
  '''
  For jaynes integration, we cannot use conditional argparse
  '''
  dummy_args_in = [
            "-alg", alg,
            "-env", env,
            "-curiosity_alg", curiosity_alg]
  return get_args(dummy_args_in)

