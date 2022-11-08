import os
import sys
import subprocess
import time
import json
import argparse
from matplotlib import use
from six.moves import shlex_quote
import GPUtil
import torch

# Runners
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval

# Policies
from rlpyt.agents.pg.atari import AtariFfAgent, AtariLstmAgent
from rlpyt.agents.pg.mujoco import MujocoFfAgent, MujocoLstmAgent

# Samplers
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector, CpuWaitResetCollector, CpuEvalCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector, GpuWaitResetCollector, GpuEvalCollector
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

# Environments
from rlpyt.samplers.collections import TrajInfo
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.envs.mazeworld.mazeworld.envs.pycolab_env import PycolabTrajInfo
from rlpyt.envs.gym import make as gym_make
from rlpyt.envs.gym import mario_make, deepmind_make

# Learning Algorithms
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C

# Utils
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.affinity import make_affinity, encode_affinity, affinity_from_code
from rlpyt.utils.launching.arguments import get_args
from rlpyt.utils.misc import wrap_print

with open('./global.json') as global_params:
    params = json.load(global_params)
    _WORK_DIR = params['local_workdir']
    _RESULTS_DIR = params['local_resultsdir']
    _TB_PORT = params['tb_port']
    _ATARI_ENVS = params['envs']['atari_envs']
    _PYCOLAB_ENVS = params['envs']['pycolab_envs']
    _MUJOCO_ENVS = params['envs']['mujoco_envs']


def launch_tmux(args):

    # determine log directory and argument string
    if args.pretrain != "None":
        log_dir = os.path.join(_RESULTS_DIR, args.pretrain)
        cmd_file = open(log_dir + '/cmd.txt')
        args_string = cmd_file.read().split(' ')
        args_string[args_string.index('-pretrain') + 1] = args.pretrain
        args_string = args_string[2:] # take out python3 launch.py
        args_string = ' '.join(args_string)
    else:
        name = '_'.join([args.alg, args.env])
        if os.path.isdir(f'{_RESULTS_DIR}/{name}/run_0'):
            runs = os.listdir(f'{_RESULTS_DIR}/{name}')
            try:
                runs.remove('tmp')
            except ValueError:
                pass
            try:
                runs.remove('.DS_Store')
            except ValueError:
                pass
            sorted_runs = sorted(runs, key=lambda run: int(run.split('_')[-1]))
            run_id = int(sorted_runs[-1].split('_')[-1]) + 1
        else:
            run_id = 0
            os.makedirs(os.path.join(_RESULTS_DIR, name, f'run_{run_id}'))
        log_dir = os.path.join(_RESULTS_DIR, name, f'run_{run_id}')

        args_string = ''
        for arg, value in vars(args).items():
            if arg == 'launch_tmux':
                args_string += '-launch_tmux no '
            elif arg == 'enemy_reward':
                args_string += '-enemy_reward {} '.format(format(value, 'f'))
            elif arg == 'obj_reward':
                args_string += '-obj_reward {} '.format(format(value, 'f'))
            elif value is None and arg == 'log_dir':
                args_string += f'-log_dir {log_dir} '
            elif value is True:
                args_string += f'-{arg} '
            elif value is False:
                pass
            else:
                args_string += f'-{arg} {value} '

    # check whether to run
    print('\n')
    print('#'*50)
    print('Generated command:')
    print('-'*50)
    print(f'python3 launch.py {args_string}')
    print('#'*50)
    print('\n')

    commands = {'htop' : 'htop',
                'tb' : f'tensorboard --logdir {log_dir} --port {_TB_PORT} --bind_all',
                'runner' : f'python3 launch.py {args_string}'}
    os.system(f'kill -9 $( lsof -i:{_TB_PORT} -t ) > /dev/null 2>&1')
    os.system('tmux kill-session -t experiment')
    os.system('tmux new-session -s experiment -n htop -d bash')
    i = 0
    for name, cmd in commands.items():
        if name != 'htop':
            os.system(f'tmux new-window -t experiment:{i+1} -n {name} bash')
        os.system(f'tmux send-keys -t experiment:{name} {shlex_quote(cmd)} Enter')
        i += 1

    # save arguments, and command if needed
    if args.pretrain == "None":
        time.sleep(6) # wait for logdir to be created
        with open(log_dir + '/cmd.txt', 'w') as cmd_file:
            cmd_file.writelines(commands['runner'])


def start_experiment(args, use_jaynes=False):
    global _RESULTS_DIR

    args_json = json.dumps(vars(args), indent=4)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    with open(args.log_dir + '/arguments.json', 'w') as jsonfile:
        jsonfile.write(args_json)

    # W&B logging
    if args.use_wandb:
      import wandb
      wandb.init(project="curiosity",
            entity="improbableai_zwh",
            config = vars(args),
            name=args.log_dir,
            sync_tensorboard=True)

    '''
    Overwrite the _RESULT_DIR so that we can set result dir from the launching script
    '''
    if args.result_dir:
      _RESULTS_DIR = args.result_dir

    if not use_jaynes:
      '''
      Jaynes monut code is not a git directory, but jaynes provides its own git record.
      '''
      with open(args.log_dir + '/git.txt', 'w') as git_file:
          branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
          commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
          git_file.write('{}/{}'.format(branch, commit))

    config = dict(env_id=args.env)

    if args.sample_mode == 'gpu':
        assert args.num_gpus > 0
        if args.gpu_id == 'auto':
          gpu_id = GPUtil.getFirstAvailable(order = 'memory', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)[0]
        else:
          gpu_id = args.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        affinity = dict(cuda_idx=0, workers_cpus=list(range(args.num_cpus)))
    else:
        affinity = dict(workers_cpus=list(range(args.num_cpus)))

    # potentially reload models
    initial_optim_state_dict = None
    initial_model_state_dict = None
    initial_algo_state_dict = None
    if args.pretrain != 'None':
        os.system(f"find {args.log_dir} -name '*.json' -delete") # clean up json files for video recorder
        try:
            checkpoint = torch.load(os.path.join('./results', args.pretrain, 'params.pkl'))
        except:
            checkpoint = torch.load(os.path.join('./results', args.pretrain, 'params_old.pkl'))
        initial_optim_state_dict = checkpoint['optimizer_state_dict']
        initial_model_state_dict = checkpoint['agent_state_dict']
        initial_algo_state_dict = checkpoint.get('algo_state_dict', None) # NOTE: backward compatiable with checkpoints without algo_state_dict

    encoder_dict = None
    encoder_pretrain = getattr(args, 'encoder_pretrain', 'None')
    if encoder_pretrain != 'None':
        try:
            checkpoint = torch.load(os.path.join('./results', encoder_pretrain, 'params.pkl'))
        except:
            checkpoint = torch.load(os.path.join('./results', encoder_pretrain, 'params_old.pkl'))
        agent_dict = checkpoint['agent_state_dict']
        encoder_dict = {'.'.join(k.split('.')[2:]):agent_dict[k] for k in agent_dict if 'encoder' in k}

    # ----------------------------------------------------- POLICY ----------------------------------------------------- #
    model_args = dict(curiosity_kwargs=dict(curiosity_alg=args.curiosity_alg))
    if args.curiosity_alg == 'icm':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['forward_model'] = args.forward_model
        model_args['curiosity_kwargs']['feature_space'] = args.feature_space
        model_args['curiosity_kwargs']['encoder_dict'] = encoder_dict
        model_args['curiosity_kwargs']['fix_features'] = args.fix_features
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
        model_args['curiosity_kwargs']['use_minmax'] =  args.use_minmax
    elif args.curiosity_alg == 'micm':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['forward_model'] = args.forward_model
        model_args['curiosity_kwargs']['ensemble_mode'] = args.ensemble_mode
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
        model_args['curiosity_kwargs']['use_minmax'] =  args.use_minmax
    elif args.curiosity_alg == 'disagreement':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['ensemble_size'] = args.ensemble_size
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['forward_model'] = args.forward_model
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
        model_args['curiosity_kwargs']['use_minmax'] =  args.use_minmax
    elif args.curiosity_alg == 'ndigo':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['pred_horizon'] = args.pred_horizon
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
        model_args['curiosity_kwargs']['use_minmax'] =  args.use_minmax
    elif args.curiosity_alg == 'rnd':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['drop_probability'] = args.drop_probability
        model_args['curiosity_kwargs']['gamma'] = args.discount_ri
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
        model_args['curiosity_kwargs']['use_minmax'] =  args.use_minmax

    if args.env in _MUJOCO_ENVS:
        if args.lstm:
            agent = MujocoLstmAgent(initial_model_state_dict=initial_model_state_dict)
        else:
            agent = MujocoFfAgent(initial_model_state_dict=initial_model_state_dict)
    else:
        if args.lstm:
            agent = AtariLstmAgent(
                        initial_model_state_dict=initial_model_state_dict,
                        model_kwargs=model_args,
                        no_extrinsic=args.no_extrinsic,
                        dual_policy=args.dual_policy,
                        use_minmax=args.use_minmax,
                        )
        else:
            agent = AtariFfAgent(initial_model_state_dict=initial_model_state_dict)

    # ----------------------------------------------------- LEARNING ALG ----------------------------------------------------- #
    if args.alg == 'ppo':
        algo = PPO(
                discount=args.discount,
                discount_ri=getattr(args, 'discount_ri', 0.0),
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict, # is None is not reloading a checkpoint
                gae_lambda=args.gae_lambda,
                minibatches=args.minibatches, # if recurrent: batch_B needs to be at least equal, if not recurrent: batch_B*batch_T needs to be at least equal to this
                epochs=args.epochs,
                ratio_clip=args.ratio_clip,
                linear_lr_schedule=args.linear_lr,
                normalize_advantage=args.normalize_advantage,
                normalize_reward=args.normalize_reward,
                normalize_extreward=args.normalize_extreward,
                normalize_intreward=args.normalize_intreward,
                rescale_extreward=args.rescale_extreward,
                rescale_intreward=args.rescale_intreward,
                dual_value=getattr(args, 'dual_value', False),
                dual_policy=args.dual_policy,
                dual_policy_noint=args.dual_policy_noint,
                dual_policy_weighting=args.dual_policy_weighting,
                dpw_formulation=args.dpw_formulation,
                utility_noworkers=args.utility_noworkers,
                kl_lambda=args.kl_lambda,
                kl_clamp=args.kl_clamp,
                util_clamp=args.util_clamp,
                util_detach=args.util_detach,
                kl_detach=args.kl_detach,
                importance_sample=args.importance_sample,
                curiosity_type=args.curiosity_alg,
                use_minmax=args.use_minmax,
                minmax_alpha=args.minmax_alpha,
                use_adapt_alpha=args.use_adapt_alpha,
                alpha_clip=args.alpha_clip,
                alpha_lr=args.alpha_lr,
                minmax_switch=args.minmax_switch,
                initial_algo_state_dict=initial_algo_state_dict,
                )
    elif args.alg == 'a2c':
        algo = A2C(
                discount=args.discount,
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict,
                gae_lambda=args.gae_lambda,
                normalize_advantage=args.normalize_advantage
                )

    # ----------------------------------------------------- SAMPLER ----------------------------------------------------- #

    # environment setup
    traj_info_cl = TrajInfo # environment specific - potentially overriden below
    if 'mario' in args.env.lower():
        env_cl = mario_make
        env_args = dict(
            game=args.env,
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=args.normalize_obs,
            normalize_obs_steps=10000
            )
    elif args.env in _PYCOLAB_ENVS:
        env_cl = deepmind_make
        traj_info_cl = PycolabTrajInfo
        env_args = dict(
            game=args.env,
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=args.normalize_obs,
            normalize_obs_steps=10000,
            log_heatmaps=args.log_heatmaps,
            logdir=args.log_dir,
            obs_type=args.obs_type,
            grayscale=args.grayscale,
            max_steps_per_episode=args.max_episode_steps
            )
    elif args.env in _MUJOCO_ENVS:
        env_cl = gym_make
        env_args = dict(
            id=args.env,
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=False,
            normalize_obs_steps=10000
            )
    elif args.env in _ATARI_ENVS:
        env_cl = AtariEnv
        traj_info_cl = AtariTrajInfo
        env_args = dict(
            game=args.env,
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=args.normalize_obs,
            normalize_obs_steps=10000,
            downsampling_scheme='classical',
            record_freq=args.record_freq,
            record_dir=args.log_dir,
            horizon=args.max_episode_steps,
            score_multiplier=args.score_multiplier,
            repeat_action_probability=args.repeat_action_probability,
            fire_on_reset=args.fire_on_reset
            )

    if args.sample_mode == 'gpu':
        if args.lstm:
            collector_class = GpuWaitResetCollector
        else:
            collector_class = GpuResetCollector
        sampler = GpuSampler(
            EnvCls=env_cl,
            env_kwargs=env_args,
            eval_env_kwargs=env_args,
            batch_T=args.timestep_limit,
            batch_B=args.num_envs,
            max_decorrelation_steps=0,
            TrajInfoCls=traj_info_cl,
            eval_n_envs=args.eval_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_traj,
            record_freq=args.record_freq,
            log_dir=args.log_dir,
            CollectorCls=collector_class
        )
    else:
        if args.lstm:
            collector_class = CpuWaitResetCollector
        else:
            collector_class = CpuResetCollector
        sampler = CpuSampler(
            EnvCls=env_cl,
            env_kwargs=env_args,
            eval_env_kwargs=env_args,
            batch_T=args.timestep_limit, # timesteps in a trajectory episode
            batch_B=args.num_envs, # environments distributed across workers
            max_decorrelation_steps=0,
            TrajInfoCls=traj_info_cl,
            eval_n_envs=args.eval_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_traj,
            record_freq=args.record_freq,
            log_dir=args.log_dir,
            CollectorCls=collector_class
            )
        affinity["master_torch_threads"] = torch.get_num_threads()

    # ----------------------------------------------------- RUNNER ----------------------------------------------------- #
    if args.eval_envs > 0:
        runner = MinibatchRlEval(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=args.log_dir,
            pretrain=args.pretrain
            )
    else:
        runner = MinibatchRl(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=args.log_dir,
            pretrain=args.pretrain,
            model_save_freq=args.model_save_freq
            )

    with logger_context(args.log_dir, config, snapshot_mode="last", use_summary_writer=True):
        runner.train()

    if args.use_wandb:
      wandb.finish()