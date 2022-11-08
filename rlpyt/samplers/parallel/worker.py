
import os
import json
import psutil
import time
import torch

from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import logger
from rlpyt.utils.seed import set_seed, set_envs_seeds

from gym.wrappers import Monitor

# root_path = os.path.abspath(__file__).split('/')[1:]
# root_path = root_path[:root_path.index('curiosity_baselines')+1]
# info_file_path = '/'+ '/'.join(root_path) + '/global.json'
info_file_path = './global.json' # Assume the global.json is at the current directory
with open(info_file_path) as global_params_file:
    global_params = json.load(global_params_file)
    ATARI_ENVS = global_params['envs']['atari_envs']

def initialize_worker(rank, seed=None, cpu=None, torch_threads=None):
    """Assign CPU affinity, set random seed, set torch_threads if needed to
    prevent MKL deadlock.
    """
    log_str = f"Sampler rank {rank} initialized"
    cpu = [cpu] if isinstance(cpu, int) else cpu
    p = psutil.Process()
    try:
        if cpu is not None:
            p.cpu_affinity(cpu)
        cpu_affin = p.cpu_affinity()
    except AttributeError:
        cpu_affin = "UNAVAILABLE MacOS"
    log_str += f", CPU affinity {cpu_affin}"
    torch_threads = (1 if torch_threads is None and cpu is not None else torch_threads)  # Default to 1 to avoid possible MKL hang.
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    log_str += f", Torch threads {torch.get_num_threads()}"
    if seed is not None:
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed {seed}"
    logger.log(log_str)


def sampling_process(common_kwargs, worker_kwargs):
    """Target function used for forking parallel worker processes in the
    samplers. After ``initialize_worker()``, it creates the specified number
    of environment instances and gives them to the collector when
    instantiating it.  It then calls collector startup methods for
    environments and agent.  If applicable, instantiates evaluation
    environment instances and evaluation collector.

    Then enters infinite loop, waiting for signals from master to collect
    training samples or else run evaluation, until signaled to exit.
    """
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)

    envs = [c.EnvCls(**c.env_kwargs) for _ in range(w.n_envs)]

    log_heatmaps = c.env_kwargs.get('log_heatmaps', None)

    if log_heatmaps is not None and log_heatmaps == True:
        for env in envs[1:]:
            env.log_heatmaps = False

    if c.record_freq > 0:
        if c.env_kwargs['game'] in ATARI_ENVS:
            envs[0].record_env = True
            envs[0].set_episode_num()
        elif c.get("eval_n_envs", 0) == 0: # only record workers if no evaluation processes are performed
            envs[0] = Monitor(envs[0], c.log_dir + '/videos', video_callable=lambda episode_id: episode_id%c.record_freq==0)

    set_envs_seeds(envs, w.seed)

    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agent", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
        global_B=c.get("global_B", 1),
        env_ranks=w.get("env_ranks", None),
        no_extrinsic=c.no_extrinsic
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    collector.start_agent()

    if c.get("eval_n_envs", 0) > 0:
        eval_envs = [c.EnvCls(**c.eval_env_kwargs) for _ in range(c.eval_n_envs)]
        if c.record_freq > 0:
            eval_envs[0] = Monitor(eval_envs[0], c.log_dir + '/videos', video_callable=lambda episode_id: episode_id%c.record_freq==0)
        set_envs_seeds(eval_envs, w.seed)
        eval_collector = c.eval_CollectorCls(
            rank=w.rank,
            envs=eval_envs,
            TrajInfoCls=c.TrajInfoCls,
            traj_infos_queue=c.eval_traj_infos_queue,
            max_T=c.eval_max_T,
            agent=c.get("agent", None),
            sync=w.get("sync", None),
            step_buffer_np=w.get("eval_step_buffer_np", None),
        )
    else:
        eval_envs = list()

    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            logger.log('Quitting worker ...')
            break
        if ctrl.do_eval.value:
            eval_collector.collect_evaluation(ctrl.itr.value)  # Traj_infos to queue inside.
        else:
            agent_inputs, traj_infos, completed_infos = collector.collect_batch(agent_inputs, traj_infos, ctrl.itr.value)
            for info in completed_infos:
                c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs + eval_envs:
        logger.log('Stopping env ...')
        env.close()
