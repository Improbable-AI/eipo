import os
import itertools

SLURM_TASK_BASE_SCRIPT = "mpirun -n 1 python3 launch.py {args}"

TIG_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --exclude=tig-slurm-4

{job_cmds}
wait < <(jobs -p)
"""

TIG_SLURM_CPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --exclude=tig-slurm-4

{job_cmds}
wait < <(jobs -p)
"""

IMP_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=imp
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

{job_cmds}
wait < <(jobs -p)
"""

SUPERCLOUD_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=normal
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --mem-per-cpu=10000M
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

module load anaconda/2021a
module load cuda/11.4
module load mpi/openmpi-4.0

{job_cmds}
wait < <(jobs -p)
"""

BASE_LOCAL_SCRIPT = """mpirun -n 1 python3 launch.py {args}"""

def exp_id():
  import uuid
  return uuid.uuid4()

def hostname():
  import subprocess
  cmd = 'hostname -f'
  try:
      p = subprocess.check_output(cmd, shell=True)  # Save git diff to experiment directory
      return p.decode('utf-8').strip()
  except subprocess.CalledProcessError as e:
      print(f"can not get obtain hostname via `{cmd}` due to exception: {e}")
      return None


def _screen(job_name, run_cmd, verbose=True):
    cmd = f"screen -S {job_name} -dm bash -c \"{run_cmd}\""
    if verbose:
      print(cmd)
    os.system(cmd)

def sbatch(job_name, args, gpu=0, parallel=True, verbose=False, **kwargs):
    task_script = ""
    for idx, task_args in enumerate(args):
      task_script += SLURM_TASK_BASE_SCRIPT.format(args=task_args) + (" &" if parallel else "")
      if idx != len(args) - 1:
        task_script += "\n"

    _hostname = hostname()
    if _hostname.startswith("slurm-control"):
      script = TIG_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
      print("Use TIG Slurm script")
    elif _hostname.startswith("improbable"):
      script = IMP_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
      print("Use improbable script")
    elif _hostname.startswith("login"):
      script = SUPERCLOUD_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
      print("Use supercloud script")

    cmd = f'/bin/bash -c \"sbatch <<< \\"{script}\\" "'
    if verbose:
      print(cmd)
    os.system(cmd)

def screen(job_name, args, verbose=False, **kwargs):
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"screen -S {job_name} -dm bash -c \"echo $STY; source setup.sh; conda activate curiosity; {script}\""
    if verbose:
      print(cmd)
    os.system(cmd)

def bash(job_name, args, gpu=None, verbose=False, **kwargs):
    assert len(args) == 1 # Not support parallel screen jobs for now
    script = BASE_LOCAL_SCRIPT.format(args=args[0])
    gpu_setup = f"CUDA_VISIBLE_DEVICES={gpu}; " if gpu else ""
    cmd = f"{gpu_setup}{script}"

    print(cmd)
    return cmd

def local(job_name, args, gpu=None, verbose=False, **kwargs):
    assert len(args) == 1 # Not support parallel screen jobs for now
    script = BASE_LOCAL_SCRIPT.format(args=args[0])
    gpu_setup = f"CUDA_VISIBLE_DEVICES={gpu}; " if gpu else ""
    cmd = f"{gpu_setup}{script}"

    if verbose:
      print(cmd)
    os.system(cmd)

def launch(job_name, args, mode, verbose=False, **kwargs):
  if mode == 'sbatch':
    sbatch(job_name, args, verbose=verbose, **kwargs)
  elif mode == 'screen':
    screen(job_name, args, verbose=verbose, **kwargs)
  elif mode == 'bash':
    return bash(job_name, args, verbose=verbose, **kwargs)
  elif mode == 'local':
    local(job_name, args, verbose=verbose, **kwargs)
  else:
    raise NotImplemented()


def get_launch_args(experiment):
  import argparse
  parser = argparse.ArgumentParser(description=f'{experiment}')
  parser.add_argument('--gpus', nargs="+", type=int, help="GPU Id lists (e.g., 0,1,2,3)", default=0)
  parser.add_argument('--mode', type=str, choices=['sbatch', 'screen', 'bash', 'local'], required=True)
  parser.add_argument('--n_parallel_task', type=int, default=1, help="Number of parallel jobs in on sbatch submission")
  parser.add_argument('--task_id', help="e.g., 5:10", type=str, required=False, default=":")
  args = parser.parse_args()
  args.gpus = [args.gpus] if isinstance(args.gpus, int) else args.gpus
  return args

def to_tuple_list(list_tuple):
  tuple_lists = [[] for i in range(len(list_tuple[0]))]
  for t in list_tuple:
    for i, e in enumerate(t):
      tuple_lists[i].append(e)
  return tuple_lists

def sweep(sweep_args, n_parallel_task=1):
  buffer = [] # a list of tuple, each tuple is one arg combination
  for args in sweep_args:
    buffer.append(args)
    if len(buffer) == n_parallel_task:
      yield (len(buffer), *to_tuple_list(buffer))
      buffer = []
  if len(buffer) > 0:
    yield (len(buffer), *to_tuple_list(buffer))
    buffer = []