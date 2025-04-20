import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from utils import *
import random

class SchedEnv(gym.Env):
    def __init__(self, env_config, type_tasks = "good_scaling"):
        self.N = env_config["N"]
        self.M = env_config["M"]
        self.type_tasks = type_tasks
        # On-Hot encoding of the observation space
        partition_space = [16]
        ready_task_space = ([(self.M + 1)] * 5 + [(self.N + 1)]) * self.N
        slices_t_space = [(self.M + 1)] * 7
        self.observation_space = MultiDiscrete(partition_space + ready_task_space + slices_t_space)
        self.action_space = Discrete(1 + 16 + 7 * self.N)


    def _get_action_mask(self):
        current_partition = self.obs["partition"]
        slices_t = self.obs["slices_t"]
        ready_tasks = self.obs["ready_tasks"]

        # The wait action is only valid if there are tasks on the GPU
        wait = 1 if any(slice_t != 0 for slice_t in slices_t) else 0

        # Valid reconfigurations
        # If there are no ready tasks or it has just been reconfigured, it doesn't make sense to reconfigure the GPU
        if ready_tasks[0][-1] == 0:
            reconfig_mask = [0] * 16
        else:
            reconfig_mask = [1] * 16
            # Forbidden to reconfigure to the current partition
            reconfig_mask[current_partition - 1] = 0
            # Forbidden to reconfigure to partitions where slices in use need to be changed
            for future_partition in range(1, 17):
                for curr_slice in range(7):
                    if partition_map[current_partition]["slices"][curr_slice] != partition_map[future_partition]["slices"][curr_slice] and slices_t[curr_slice] > 0:
                        reconfig_mask[future_partition - 1] = 0

        # Valid actions on ready tasks
        select_ready_task = [1] * 7 * self.N
        # It is prohibited to select a task to execute on the GPU beyond the available types
        for i, task in enumerate(ready_tasks):
            n_availables = task[-1]
            # If there is an unavailable task in ready, all subsequent ones are unavailable due to canonical order
            if n_availables == 0:
                for j in range(i*7, 7 * self.N):
                    select_ready_task[j] = 0
                break
        
        # It is prohibited to assign the ready task to an instance with an index higher than the number of instances in the current partition
        first_forbidden_instance = len(partition_map[current_partition]["sizes"])
        for task in range(self.N):
            for instance in range(first_forbidden_instance, 7):
                select_ready_task[task * 7 + instance] = 0

        # It is prohibited to assign a task to an instance with a task already running (or reconfiguring)
        first_instance_slice = 0
        for instance_index in range(first_forbidden_instance):
            if slices_t[first_instance_slice] > 0:
                for task in range(self.N):
                    select_ready_task[task * 7 + instance_index] = 0
            current_size = partition_map[current_partition]["sizes"][instance_index]
            if current_size == 3 and instance_index == 0:
                current_size = 4
            first_instance_slice += current_size

        return [wait] + reconfig_mask + select_ready_task
        


    def get_numpy_obs_state(self):
        obs = [self.obs["partition"] - 1]
        for task in self.obs["ready_tasks"]:
            obs += task
        obs += self.obs["slices_t"]
        
        return (np.array(obs, dtype=np.float32)/max(self.M + 1, self.N, 19))

    def valid_action_mask(self):
        return np.array(self._get_action_mask())
    


    def reset(self, seed = None, options = None):
        # Randomly select the initial partition
        init_partition = random.randint(1, 16)
        # Randomly select the initial time for each slice
        part_sizes = partition_map[init_partition]["sizes"]
        init_slice_t = []
        for instance_size in part_sizes:
            init_instance_time = random.randint(0, self.M)
            init_slice_t += [init_instance_time] * instance_size

        # Keeps the number of task types running on each slice
        self.num_task_slices = partition_map[init_partition]["instances"].copy() 
        
        # Keeps the number of task types running on each slice
        ready_tasks = get_ready_tasks(self.type_tasks, self.N)
        ready_tasks, self.reconfig_map_scaled = time_discretization(ready_tasks, self.M)
        ready_tasks_canonical, self.dic_cont_times = canonical_sort_tasks(self.M, ready_tasks)
        

        # To have an index with the task type number, which then allows consistency in the graphical representation
        self.num_type_task = list(range(len(ready_tasks_canonical)))
        # Use 0 to fill empty positions in the ready task representation
        init_ready_tasks = ready_tasks_canonical + [[0] * 6] * (self.N - len(ready_tasks_canonical)) # Fill with arrays of 6 zeros up to N
        self.obs = {
            "partition": init_partition,
            "ready_tasks": init_ready_tasks,
            "slices_t": init_slice_t,
        }
        self.obs["action_mask"] = self._get_action_mask()

        self._check_obs_consistency()

        self.last_action = None # No action has been taken yet

        self.acum_reward = 0
        
        self.init_state = {"partition": init_partition, "slices_t": init_slice_t}
        self.actions = []
        
        return self.get_numpy_obs_state(), {}
    

    def render(self):
        basic_print_obs(self.obs)

    def _is_terminated(self):
        # If all tasks are finished, and everything on the GPU is done, the episode ends
        for slice_t in self.obs["slices_t"]:
            if slice_t > 0:
                return False

        for ready_task in self.obs["ready_tasks"]:
            if ready_task[-1] > 0:
                return False
        return True
    
    def _check_obs_consistency(self):
        times = {}
        for i, instance_num in enumerate(partition_map[self.obs["partition"]]["instances"]):
            if instance_num not in times:
                times[instance_num] = self.obs["slices_t"][i]
            else:
                if times[instance_num] != self.obs["slices_t"][i]:
                    pprint(self.actions)
                    print(self.obs["partition"], self.obs["slices_t"])
                assert times[instance_num] == self.obs["slices_t"][i] # All slices of the same instance must have the same time
    

    def step(self, action):
        current_partition = self.obs["partition"]
        slices_t = self.obs["slices_t"]
        ready_tasks = self.obs["ready_tasks"]
        
        # Wait
        if action == 0:
            # Transition to the first slice that is freed
            min_slice_time = min(slice_time for slice_time in slices_t if slice_time > 0)
            
            self.obs["slices_t"] = [slice_time - min_slice_time if slice_time > 0 else 0 for slice_time in slices_t]
            # Reward with -elapsed time, to minimize makespan
            reward = -min_slice_time
            self.actions.append(("wait", None))
            
        # Reconfigure
        elif action <= 16:
            next_partition = int(action)            
            # Increase time associated with the reconfiguration of the slices
            for i, instance_size in enumerate(partition_map[self.obs["partition"]]["slices"]):
                old_instance_size = int(instance_size)
                new_instance_size = int(partition_map[next_partition]["slices"][i])
                # Instance to destroy
                if old_instance_size != new_instance_size:
                    # Increase time for the instance to destroy
                    self.obs["slices_t"][i] += self.reconfig_map_scaled[old_instance_size]["destroy"]
                    # Increase time for the instance to create
                    self.obs["slices_t"][i] += self.reconfig_map_scaled[new_instance_size]["create"]
            
            
            # Basis change (equivalent reconfigs)
            if next_partition == 11 or next_partition == 12 or next_partition == 13: 
                slice_0, slice_1 = self.obs["slices_t"][0], self.obs["slices_t"][1]
                self.obs["slices_t"][0], self.obs["slices_t"][1] = self.obs["slices_t"][2], self.obs["slices_t"][3]
                self.obs["slices_t"][2], self.obs["slices_t"][3] = slice_0, slice_1
                next_partition -= 3
                self.actions.append(("exchange", None))
            # Set the new partition
            self.obs["partition"] = next_partition
            
            reward = 0
            self.actions.append(("reconfig", next_partition))
        # Assign task
        else:
            task = (action - 17) // 7
            instance = (action - 17) % 7
            # Increase the time it takes for the task for the instance size in all slices of the instance
            instance_size = partition_map[current_partition]["sizes"][instance]

            # Mark the task as selected in the list of performed actions
            cont_time_selected = self.dic_cont_times[type_num_task(self.M, self.obs["ready_tasks"][task][:5])][0][instance_size_map[instance_size]]
            self.actions.append(("assign", (cont_time_selected, instance)))
            self.dic_cont_times[type_num_task(self.M, self.obs["ready_tasks"][task][:5])].pop(0)

            task_time = ready_tasks[task][instance_size_map[instance_size]]
            # Remove the task from ready_tasks
            self.obs["ready_tasks"][task][-1] -= 1

            for i, instance_slice in enumerate(partition_map[current_partition]["instances"]):
                if instance_slice == instance:
                    self.obs["slices_t"][i] = task_time
                    self.num_task_slices[i] = self.num_type_task[task]

            # If it is the last of a certain type, remove the type from ready tasks
            if ready_tasks[task][-1] == 0:
                ready_tasks.pop(task)
                # Add an empty task type at the end to maintain dimensionality
                ready_tasks.append([0] * 6)
                # Move that task type number to the end
                self.num_type_task.append(self.num_type_task[task])
                self.num_type_task.pop(task)
            reward = 0

        self._check_obs_consistency()

        # Update the action mask for the new state
        self.obs["action_mask"] = self._get_action_mask()

        # Check if the episode has ended
        terminated = self._is_terminated()
        
        truncated, info = False, {}

        self.last_action = action # The last action performed is the one just taken

        self.acum_reward += reward

        return self.get_numpy_obs_state(), reward, terminated, truncated, info

    def close(*args, **kwargs):
        pass