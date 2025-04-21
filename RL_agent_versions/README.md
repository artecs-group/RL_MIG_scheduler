# Training tool
Modeling of different versions of a Deep Reinforcement Learning agent that optimizes task scheduling on a GPU with MIG technology. Includes training scripts using the PPO algorithm with action masking.
## RL agent versions
Each directory contains a different version of the agent (see the [paper]() for more details):

- 📁 [one_hot](https://github.com/Jorgitou98/RL_MIG_scheduler/tree/main/RL_agent_versions/one_hot): The discretized duration of the tasks is represented by One-Hot encoding in the agent's neural network input.
- 📁 [float](https://github.com/Jorgitou98/RL_MIG_scheduler/tree/main/RL_agent_versions/float): Task duration times are discrete but represented as real numbers in the agent input rather than as a One-Hot vector.
- 📁 [entropy](https://github.com/Jorgitou98/RL_MIG_scheduler/tree/main/RL_agent_versions/float): Model with *float* representation but higher entropy coefficient to favor exploration over exploitation, especially in the early stages of training. Entropy decays gradually in steps during training.
- 📁 [direct_reconfig](https://github.com/Jorgitou98/RL_MIG_scheduler/tree/main/RL_agent_versions/direct_reconfig): Evolution of the *entropy* version, allowing reconfiguration in any state to facilitate the exploration of such actions.
- 📁 [online](https://github.com/Jorgitou98/RL_MIG_scheduler/tree/main/RL_agent_versions/online): The other versions follow an *offline* scheme where tasks are handled in separate batches without replenishment of new tasks. This version follows an *online* approach, where tasks are replenished in the batch when their size allows it. See Section 3 of the paper for more details.

## Configuration setup
The models have been developed with Python 3.11.0, but Python >= 3.8 should be sufficient. We recommend installing the necessary dependencies, specified in the `requirements.txt` file, in a Python virtual environment common to all models as follows:

```bash
python -m venv venv
source venv/bin/activate  # If you are with Linux / macOS (.\venv\Scripts\activate   # If you are with Windows)
pip install -r requirements.txt
```

## Usage
Each model consists of a ``train.py`` script for training. This script can receive the following arguments:

- ``--num_steps``: Number of training timesteps for the model. If not specified, it defaults to 200 million.
- ``--N``: Number of tasks in the batch handled by the agent for scheduling. It is one of the dimension parameters of the agent's observations. By default, it takes the value 14.
- ``--M``: Number of levels in the discretization of the agent. This is the other parameter on which the agent's observations depend. By default, it takes the value 14.
- ``--device``: Type of device where the training is to be performed. Possible values are "gpu" or "cpu". To use a certain GPU of your system set the integer variable CUDA_VISIBLE_DEVICES in the script invocation (an example is shown below). By default its values is "cpu".
- ``--type_tasks``: Type of workload with which to train the model. Possible values are: "good_scaling", "bad_scaling", "mix_scaling_extreme", "mix_scaling_soft" and "wide_times". The generation and characteristics of these workloads are explained in detail in the [paper](). The default value is "good_scaling".
- ``--restore``: Route to the model from which to re-establish to continue your training. If this parameter is not provided, a new model (with the characteristics specified in the paper) is created and starts training from scratch.

The following is an example of calling the training of a direct_reconfig model, specifying all parameters, using GPU number 0 and restoring from the model "trained_models/bs3_N=14_M=14_s=50000_wide_times.zip":

```bash
cd direct_reconfig/
CUDA_VISIBLE_DEVICES = 0 python train.py --num_steps 100000 --N 14 --M 14 --device gpu --type_tasks wide_times --restore trained_models/bs3_N=14_M=14_s=50000_wide_times.zip
```

## Output
During training, metrics such as average reward or training speed (in fps) are periodically displayed by the standard output. The final output is a model in zip format that is saved in the ``/trained_models`` directory within the model directory. The name of this model includes the ``N``, ``M``, ``num_steps`` and ``type-tasks`` parameters of the training. For the above example:
```
trained_models/bs3_N=14_M=14_s=100000_wide_times.zip
```
For safety, for long training sessions, the model is saved periodically every 10 million iterations. If the user at any time sends an interrupt signal via ctr+C the model is also saved before the end of the program.


