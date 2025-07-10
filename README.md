# RL-based scheduler to optimize the use of Multi-Instance GPU (MIG)

This repository contains three tools to optimize co‑execution of workloads on NVIDIA GPUs using the dynamic partitioning features of Multi‑Instance GPU (MIG) technology.
The modeling approach and experimental results for these tools are detailed in the [associated paper]().
Each tool has its own README —with requirements, installation instructions, functionality, and usage— located in the directories linked below:

## Tools Overview
- **Training tool :file_folder: [RL_agent_versions/](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/RL_agent_versions)**  
  Provides the modeling and scripts to train a MIG task scheduler using the PPO Deep Reinforcement Learning algorithm. It includes multiple agent versions refined in different ways, all described in detail in [the paper]().

- **Visualization tool :file_folder: [visual_scheduler/](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/visual_scheduler)**  
  Plots and allows to interact with the agent’s observations and decisions for a given workload. This is invaluable for debugging, understanding, and illustrating the agent’s behavior in specific scenarios.

- **MIG Scheduling tool :file_folder: [GPU_MIG_scheduler/](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/GPU_MIG_scheduler)**  
  Enables optimized co‑execution of a set of tasks on a MIG‑capable GPU by following the decision policy of a pre‑trained agent from the Training tool. This tool also includes the FAR scheduler, presented in a [previous paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4958466), which can be used as an alternative to the RL agent.

The folder [aux_scripts](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/aux_scripts) contains some auxiliary scripts used for processing and plotting data for the paper.

## Requirements
  - Python 3.8 or newer
  - NVIDIA GPU with MIG support (models A30, A100, H100, B100, B200)
  - CUDA Toolkit ≥ 11.0 (including NVML library)
  - CMake ≥ 3.10
  - g++ 12.2.0 for development (robust support for C++17 should be enough)
  - Linux recommended. NVML is available on Windows, but MIG handling is limited or not supported in some cases.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/artecs-group/RL_MIG_scheduler.git
   cd RL_MIG_scheduler
   ```

2. For each tool, navigate to its directory and follow the instructions in its README:

   - [`RL_agent_versions/`](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/RL_agent_versions) → Training Agent tool
   - [`visual_scheduler/`](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/visual_scheduler) → Visualization tool
   - [`GPU_MIG_scheduler/`](https://github.com/artecs-group/RL_MIG_scheduler/tree/main/GPU_MIG_scheduler) → MIG Scheduler tool

## License

This project is licensed under the **MIT License**. See [`LICENSE`](https://github.com/artecs-group/RL_MIG_scheduler/blob/main/LICENSE) for details.

## Acknowledgements
This work is funded by Grant PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by *ERDF A way of making Europe*.

