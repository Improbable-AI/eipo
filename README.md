

## Overview

This official codebase for ["Redeeming Intrinsic Rewards via Constrained Optimization"](https://williamd4112.github.io/pubs/neurips22_eipo.pdf) (NeurIPS'22). **Cleaner implementation in CleanRL is available here: https://github.com/williamd4112/cleanrl/blob/master/cleanrl/eipo_rnd_envpool.py**

### Usage

1.  Clone this repo.

2.  If you plan on using mujoco, place your license key "mjkey.txt" in the base directory. This file will be copied in when you start docker
using the Makefile command.

3.  Make sure you have docker installed to run the [image](https://hub.docker.com/repository/docker/echen9898/curiosity_baselines). We recommend
running the GPU image which will work even if you are only using CPUs (labeled version_gpu), but a CPU only image is provided as well.

4.  Edit global.json to customize any volume mount points, port forwarding, and docker image versions from the registry. Information from this file
is read into the Makefile.

5.  The makefile contains some basic commands (we use node to read in information from global.json at the top - it's not used for anything else).
```
make start_docker # start the docker container and drop you in a shell
make start_docker_gpu # start the docker container if running on a machine with GPUs
make stop_docker # stop the docker container and clean up
make clean # clean all subdirectories of pycache files etc.
```

6.  Before running anything, make sure you create an empty directory titled "results" in the base directory.

7.  Run experiment via the following command
```
python experiments/diff_adapt_minmax/run.py --mode local --gpus 0
```

## Notes

Our codebase is based on [curiosity_baselines](https://github.com/echen9898/curiosity_baselines) and [rlpyt](https://github.com/astooke/rlpyt) For more information on the original rlpyt codebase, please see this [white paper on Arxiv](https://arxiv.org/abs/1909.01500).

### Code Organization

The class types perform the following roles:

* **Runner** - Connects the `sampler`, `agent`, and `algorithm`; manages the training loop and logging of diagnostics.
  * **Sampler** - Manages `agent` / `environment` interaction to collect training data, can initialize parallel workers.
    * **Collector** - Steps `environments` (and maybe operates `agent`) and records samples, attached to `sampler`.
      * **Environment** - The task to be learned.
        * **Observation Space/Action Space** - Interface specifications from `environment` to `agent`.
      * **TrajectoryInfo** - Diagnostics logged on a per-trajectory basis.
  * **Agent** - Chooses control action to the `environment` in `sampler`; trained by the `algorithm`.  Interface to `model`.
    * **Model** - Torch neural network module, attached to the `agent`.
    * **Curiosity Model** - Torch neural network module, attached to the `model` which is attached to the `agent`.
    * **Distribution** - Samples actions for stochastic `agents` and defines related formulas for use in loss function, attached to the `agent`.
  * **Algorithm** - Uses gathered samples to train the `agent` (e.g. defines a loss function and performs gradient descent).
    * **Optimizer** - Training update rule (e.g. Adam), attached to the `algorithm`.
    * **OptimizationInfo** - Diagnostics logged on a per-training batch basis.

### Sources and Acknowledgements

This codebase is currently funded by [Amazon MLRA](https://www.amazon.science/research-awards) - we thank them for their support.

Parts of the following open source codebases were used to make this codebase possible. Thanks to all of them for their amazing work!

* [rlpyt](https://github.com/astooke/rlpyt)
* [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
* [pycolab](https://github.com/deepmind/pycolab)
* [stable-baselines](https://github.com/hill-a/stable-baselines)

Thanks to [Prof. Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/) and the members of the [Improbable AI lab](https://people.csail.mit.edu/pulkitag/) at MIT CSAIL for their continued guidance and support.




