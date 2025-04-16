# FAR scheduler for NVIDIA Multi-Instance GPU (MIG)
This repository contains a C++ implementation of the FAR task scheduler, targeted for NVIDIA GPUs that support physical partitioning via [Multi-Instance GPU](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) (MIG). This scheduler is useful for reducing the joint execution time of tasks (makespan) by cleverly co-executing them with MIG. The FAR algorithm used by the scheduler is presented and accompanied by a comprehensive evaluation in [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4958466) (preprint for the moment). 

## Supported GPUs
Currently, the software explicitly supports the following NVIDIA GPU models:  
- **NVIDIA A30**  
- **NVIDIA A100**  
- **NVIDIA H100**
  
Moreover, it probably works correctly for models with the same MIG characteristics as some of the above. In particular, it will probably work for models NVIDIA B100 and B200 that have the same MIG partitioning as A100 and H100, although it is not tested for them.

## Requirements
To use this software, ensure the following prerequisites are met:

#### Hardware
- An NVIDIA GPU with MIG technology (see previous section for supported models).

#### Software
- **Operating System**: Linux recommended. NVML is available on Windows, but MIG management is limited or unsupported in some cases. Use on Windows is experimental and not guaranteed.
- **NVIDIA Driver**: Version 470 or later, with MIG support enabled.
- **CUDA Toolkit**: Version 11.0 or newer.
- **NVIDIA Management Library (NVML)**: Comes bundled with the NVIDIA driver; ensure it is correctly installed (usually in `/usr/include/` or `/usr/local/cuda/include`).  

#### Build Dependencies
- **CMake**:  
  - Version 3.10 or newer to configure and build the project.
- **C++ Compiler**:  
  - g++ version at least 5.1 to robustly support C++11.  

#### Permissions
- Ensure you have administrative privileges to configure MIG instances on the GPU.

## Installation
To install and build the project, follow these steps:
#### 1. Download the repository
One option is to clone this repository on your local machine using the following command:
```bash
git clone https://github.com/artecs-group/FAR_MIG_scheduler.git
cd FAR_MIG_scheduler
```
Another option is to download it from one of the generated releases.
##### 2. Update the CUDA root directory
Edit the first line of `/FAR_MIG_scheduler/CMakeLists.txt` changing the CUDA_ROOT path to point to your CUDA installation directory.
```
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA Toolkit root directory")
```
#### 3. Generate build files with CMake and compile the project
You must be located in the root directory of the project (`FAR_MIG_scheduler`). Build it in a `FAR_MIG_scheduler/build` directory:
```bash
mkdir -p build
cd build
cmake ..
```
#### 4. Compile the project
This will create two executable files: ``mig_scheduler.exe`` and ``mig_scheduler_debug.exe``. Both do the same thing, but the debugging one includes some stop points to make it easier to follow the execution (it stops and asks the user to type some key to continue).
```bash
make
```
## Rodinia test
The scheduler is ready to be tested with 9 kernels of the [Rodinia suite](https://lava.cs.virginia.edu/Rodinia/download_links.htm). The code of Rodinia configured for that test is provided as file `gpu-rodinia.tar.gz`, attached in the releases due to its huge size (some IO files are very large). Either of these two options can be used to include it:
- Option 1:
  Run the `scripts/prepare_test_data.sh` script which will download the properly configured Rodinia kernels and data, unzip it and delete the compressed file.
  
  ```bash
  cd scripts
  sh prepare_test_data.sh
  ```
- Option 2
Download the file `gpu-rodinia.tar.gz` from some release of this repository, and unzip its content using some tool like gzip in the path `FAR_MIG_scheduler/data/kernels`.

Once this is done, the test can be run by executing the scheduler as indicated in the next section. The path `../input_test/kernels_rodinia.txt` must be pass as task file (second argument of the program).

## Usage
To use the software, invoke the `mig_scheduler.exe` executable with the following arguments:

1. **GPU Index**: Specify the index of the GPU where tasks will be executed. You can use the `nvidia-smi` command to list the GPUs available on your system and their respective indices.
2. **Task File Path**: Provide the path to a file containing information about the tasks to execute. The file should follow this format:
   - Each line represents a task and contains **three fields separated by spaces**:
     1. **Task Name**: A unique identifier for the task, used for reporting during execution.
     2. **Task Directory**: The directory path where is located the script with the task.
     3. **Task Script**: The name of the script that executes the GPU kernels, defining the task.

#### Example Usage
To schedule a set of GPU tasks from the Rodinia suite (included in this repository), you can use the following command (its important to run it with ``sudo`` administrative permissions to use MIG):

```bash
sudo ./mig_scheduler.exe 0 ../data/input_test/kernels_rodinia.txt
```
where:
- `0` specifies the GPU index (GPU 0 on the system).
- `../data/input_test/kernels_rodinia.txt` is the path to the task file.
Below is an example content of the `kernels_rodinia.txt` file with 2 tasks:
```
gaussian ../kernels/gpu-rodinia/cuda/gaussian run
pathfinder ../kernels/gpu-rodinia/cuda/pathfinder run
```
Each line describes a task. For example the first task is named `gaussian`. It is located in the directory `../kernels/gpu-rodinia/cuda/gaussian`, and the script `run` in that directory is executed to perform the task.

## Execution Report
During execution, the program provides detailed logging through standard output and error streams. These messages are prefixed with the tags `INFO` for general information and `ERROR` for critical issues. Most errors correspond to unrecoverable exceptions and terminate execution.

#### Example Execution Flow
1. **Initial GPU Information**<br>
   At the start of execution, the program reports the provided GPU name and confirms that MIG has been activated:
   
   ```
   INFO: Device has been binded
   INFO: GPU model: NVIDIA A30 detected
   INFO: MIG has been activated
   ```
2. **Profiling Times**<br>
   The program profiles reconfiguration times (creation and destruction of MIG instances) and task execution times. Note: Task profiling is currently done by executing tasks but may be replaced with faster profiling methods in future versions (see [the paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4958466) for references). Example output:
   ```
   INFO: Profiling instance creation and destruction times
   INFO: Instance(start=0, size=1) has been created
   INFO: Size 1. Creation time: 0.16s. Destruction time: 0.10s
   INFO: Instance(start=0, size=2) has been created
   INFO: Size 2. Creation time: 0.13s. Destruction time: 0.11s
   ...
   INFO: Task gaussian profiled with 22.13s in size 1
   ```
3. **Repartitioning Tree and Scheduling Plan**<br>
   The program calculates and outputs a partitioning tree that includes the scheduling plan calculated by the algorithm and the estimated execution times. These trees show the hierarchy of possible instances on the GPU being used across the nodes, storing in each of them a list with the names of the tasks to be executed (`task` attribute), along with the list of the expected completion time for them (`end` attribute). For more details on these trees see Section 3.2 of the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4958466), especially the references to Figure 4. Example of output:
   ```
    ======================================
    Node(start=0, size=4, tasks=[], ends=[])
    --Node(start=0, size=2, tasks=[lavaMD], ends=[21.72])
    ----Node(start=0, size=1, tasks=[huffman, heartwall], ends=[23.01, 23.05])
    ----Node(start=1, size=1, tasks= [nw, particlefilter], ends=[22.79, 23.23])
    --Node(start=2, size=2, tasks=[], ends=[])
    ----Node(start=2, size=1, tasks=[gaussian], ends=[22.45])
    ----Node(start=3, size=1, tasks=[pathfinder, lu], ends=[21.09, 28.86])
    Tree makespan: 28.86s
    ======================================
   ```
   In the example below, the `lavaMD` task has been assigned to the GPU instance spanning the first 2 slices (`start = 0` and `size = 2`). This task is expected to complete in `21.72s` after the execution of the entire set of tasks begins. Once it finishes, that GPU instance will be split into two instances of size 1: the first with `start = 0` and `size = 1`, and the second with `start = 1` and `size = 1`. The `huffman` and `heartwall` tasks are assigned to the first new instance and are expected to finish at `23.01s` and `23.05s`, respectively, from the start of the overall execution. Similarly, the `nw` and `particlefilter` tasks are assigned to the second GPU instance, and are expected to finish at `22.79s` and `23.23s`, respectively, from the start.
5. **Task Execution**<br>
   Tasks are executed on the GPU according to the scheduling plan, with real-time execution reports for each task, including real start and end times:
   ```
   INFO: Start task pathfinder at 0.12s
   ...
   INFO: End task pathfinder at 20.64s
   ...
   INFO: Start task huffman at 20.85s
   ...
   INFO: End task huffman at 23.54s
   ```
6. **MIG Deactivation**<br>
   After task execution is completed, the program deactivates MIG and logs the corresponding information:
   ```
   INFO: MIG has been deactivated
   ```
#### Task execution output
To facilitate the tracking of the execution through the above mentioned reports, during the execution of each task, the standard and error outputs are redirected to a file in with the path `./logs-<year>-<month>-<day>/<task name>.log`. These files, indexed by date, accumulate the output of the different executions of each task in a day. Thus, the standard output of the scheduler is much more summarized and shows the primary information of the proposal (as explained above).

## Known Issues

- **Bugs in NVIDIA A100 and H100**: In these models, the [MIG API of the NVML library](https://docs.nvidia.com/deploy/nvml-api/group__nvmlMultiInstanceGPU.html) currently has some bugs in the handling of some instances, and the scheduler may not work correctly. Most of the problems occur with the size 3 instances supported exclusively by these GPUs which, as explained in the paper, are of little importance for the scheduler. However, there are some other errors that may appear occasionally. With a future version of the library that fixes these problems the scheduler should work perfectly for these GPUs.
- **Error in the destruction of instance “device in use by another user”**: Sometimes an instance that has been released is briefly locked for a short period of time with the message “device in use by another user”. The code solves this by trying to remove it in a loop until it succeeds, usually after a few tenths of a second. This influences very slightly the actual execution of the schedule, which would be slightly improved if the library would fix this bug in the future.

## Publications
The paper presenting this scheduler is currently under review. For the moment you can access the [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4958466).
## Acknowledgements
This work is funded by Grant PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by _"ERDF A way of making Europe"_. 
