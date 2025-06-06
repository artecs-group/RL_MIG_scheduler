from collections import Counter
from pprint import pprint
import os
import random
from task_times import generate_tasks

# Mapa de número de partición a sus instancias
partition_map = {
    1: {"slices": ["7"] * 7, "sizes": [7], "instances" : [0,0,0,0,0,0,0]}, 
    2: {"slices": ["4"] * 4 + ["3"] * 3, "sizes": [4, 3], "instances" : [0,0,0,0,1,1,1]}, 
    3: {"slices": ["4"] * 4 + ["2"] * 2 + ["1"], "sizes": [4, 2, 1], "instances" : [0,0,0,0,1,1,2]}, 
    4: {"slices": ["4"] * 4 + ["1"] * 3, "sizes": [4, 1, 1, 1], "instances" : [0,0,0,0,1,2,3]}, 
    5: {"slices": ["2"] * 4 + ["3"] * 3, "sizes": [2, 2, 3], "instances" : [0,0,1,1,2,2,2]}, 
    6: {"slices": ["2"] * 6 + ["1"], "sizes": [2, 2, 2, 1], "instances" : [0,0,1,1,2,2,3]}, 
    7: {"slices": ["2"] * 4 + ["1"] * 3, "sizes": [2, 2, 1, 1, 1], "instances" : [0,0,1,1,2,3,4]}, 
    8: {"slices": ["2"] * 2 + ["1"] * 2 + ["3"] * 3, "sizes": [2, 1, 1, 3], "instances" : [0,0,1,2,3,3,3]}, 
    9: {"slices": ["2"] * 2 + ["1"] * 2 + ["2"] * 2 + ["1"], "sizes": [2, 1, 1, 2, 1], "instances" : [0,0,1,2,3,3,4]}, 
    10: {"slices": ["2"] * 2 + ["1"] * 5, "sizes": [2, 1, 1, 1, 1, 1], "instances" : [0,0,1,2,3,4,5]}, 
    11: {"slices": ["1"] * 2 + ["2"] * 2 + ["3"] * 3, "sizes": [1, 1, 2, 3], "instances" : [0,1,2,2,3,3,3]}, 
    12: {"slices": ["1"] * 2 + ["2"] * 4 + ["1"], "sizes": [1, 1, 2, 2, 1], "instances" : [0,1,2,2,3,3,4]}, 
    13: {"slices": ["1"] * 2 + ["2"] * 2 + ["1"] * 3, "sizes": [1, 1, 2, 1, 1, 1], "instances" : [0,1,2,2,3,4,5]}, 
    14: {"slices": ["1"] * 4 + ["3"] * 3, "sizes": [1, 1, 1, 1, 3], "instances" : [0,1,2,3,4,4,4]}, 
    15: {"slices": ["1"] * 4 + ["2"] * 2 + ["1"], "sizes": [1, 1, 1, 1, 2, 1], "instances" : [0,1,2,3,4,4,5]}, 
    16: {"slices": ["1"] * 7, "sizes": [1, 1, 1, 1, 1, 1, 1], "instances" : [0,1,2,3,4,5,6]}, 
}


# Mapa de tamaño de instancia a posición de array en que se codifica
instance_size_map = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4}


# Pasa a entero desde M-ário como representación
def type_num_task(M, task):
    return sum(time * (M ** i) for i, time in enumerate(task[::-1]))

def _num_task_to_times(numbered_tasks):
    dic_cont, dic_discrete = {}, {}
    for num, task in numbered_tasks:
        if num not in dic_cont:
            dic_cont[num] = [[time_c for time_c, _ in task]]
            dic_discrete[num] = [time_d for _, time_d in task]
        else:
            dic_cont[num].append([time_c for time_c, _ in task])
    return dic_cont, dic_discrete

def canonical_sort_tasks(M, tasks):
    # Pongo el número de tipo de tarea a cada una
    numbered_tasks = [(type_num_task(M, [time_d for _, time_d in task]), task) for task in tasks]
    dic_cont, dic_discrete = _num_task_to_times(numbered_tasks)
    # Ordeno por tipo de tarea
    canonical_tasks = sorted(dic_discrete.items(), key=lambda x: x[0])
    # Añado como última componente de cada tipo la cantidad de veces que se repite
    canonical_tasks = [task + [len(dic_cont[type])] for type, task in canonical_tasks]     
    return canonical_tasks, dic_cont


def basic_print_obs(obs):
    state = obs["observations"]
    action_mask = obs["action_mask"]
    print("-----------")
    print("State:")
    print("\tPartition:", partition_map[state["partition"]]["sizes"])
    print("\tSlices:", state["slices_t"])
    for task in state["ready_tasks"]:
        if task[-1] != 0:
            print("\tTask type:", task[:5], "number", task[5])
    print("Action mask:")
    print("\tEsperar:", action_mask[0])
    print("\tReconfiguración:", action_mask[1:17])
    for i, task in enumerate(state["ready_tasks"]):
        print("\tPut task in instance:", action_mask[17 + i * 7: 17 + (i+1) * 7])
    print("-----------")

def action_to_str(action):
    action = int(action)
    if action == 0:
        return "Wait"
    elif action < 17:
        return f"Reconfigure to {partition_map[action]['sizes']}"
    else:
        task = (action - 17) // 7
        instance = (action - 17) % 7
        return f"Put task {task} in instance {instance}"
    
def time_discretization(ready_tasks, M, reconfig_time):
    max_time = max(max(task) for task in ready_tasks)
    time_step = max_time / M
    reconfig_time_scaled = reconfig_time / time_step
    return [[(time, round(time / time_step) + 1) for time in task] for task in ready_tasks], reconfig_time_scaled


def _n_scale_padding(n_scale, instance_sizes, N):
    sum_scale = sum(n_scale.values())
    for ins_size in instance_sizes:
        if sum_scale >= N:
            break
        n_scale[ins_size] += 1
        sum_scale += 1
    return n_scale

def get_ready_tasks(type_tasks, N):
        instance_sizes=[1,2,3,4,7]
        if type_tasks == "good_scaling":
            scale_percs = [0,0,0.2,0.4,0.4]
            n_scale= {ins_size: int(perc*N) for ins_size, perc in zip(instance_sizes, scale_percs)}
            n_scale = _n_scale_padding(n_scale, instance_sizes[::-1], N)
            ready_tasks = generate_tasks(instance_sizes=instance_sizes, n_scale=n_scale, device="A100", perc_membound=100, times_range=[90,100])
        elif type_tasks == "bad_scaling":
            scale_percs = [0.2,0.2,0.2,0.2,0.2]
            n_scale= {ins_size: int(perc*N) for ins_size, perc in zip(instance_sizes, scale_percs)}
            n_scale = _n_scale_padding(n_scale, instance_sizes, N)
            ready_tasks = generate_tasks(instance_sizes=instance_sizes, n_scale=n_scale, device="A100", perc_membound=50, times_range=[90,100])
        elif type_tasks == "mix_scaling_extreme": 
            ready_tasks_good = get_ready_tasks(type_tasks="good_scaling", N = N // 2)
            ready_tasks_bad = get_ready_tasks(type_tasks="bad_scaling", N = N // 2 if N % 2 == 0 else N // 2 + 1)
            ready_tasks = ready_tasks_good + ready_tasks_bad
        elif type_tasks == "mix_scaling_soft":
            scale_percs = [0.2,0.2,0.2,0.2,0.2]
            n_scale= {ins_size: int(perc*N) for ins_size, perc in zip(instance_sizes, scale_percs)}
            n_scale = _n_scale_padding(n_scale, instance_sizes, N)
            ready_tasks = generate_tasks(instance_sizes=instance_sizes, n_scale=n_scale, device="A100", perc_membound=50, times_range=[90,100])
        elif type_tasks == "wide_times":
            scale_percs = [0.2,0.2,0.2,0.2,0.2]
            n_scale= {ins_size: int(perc*N) for ins_size, perc in zip(instance_sizes, scale_percs)}
            n_scale = _n_scale_padding(n_scale, instance_sizes, N)
            ready_tasks = generate_tasks(instance_sizes=instance_sizes, n_scale=n_scale, device="A100", perc_membound=50, times_range=[1,100])
            
        return ready_tasks


def makespan_lower_bound(dic_cont_times):
    pos_to_slices = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7}
    sum_min_area = 0
    for num_task, times in dic_cont_times.items():
        for times_t in times:
            areas = [pos_to_slices[pos] * time for pos, time in enumerate(times_t)]
            min_area = min(areas)
            sum_min_area += min_area
    return sum_min_area / 7

def compute_makespan(init_state, actions):
    partition = init_state["partition"]
    slices_t = init_state["slices_t"]
    for type, val in actions:
        if type == "reconfig":
            partition = val
        elif type == "assign":
            time, instance = val
            min_time_next_task = max([slices_t[pos] for pos, slice_ins in enumerate(partition_map[partition]["instances"]) if slice_ins == instance]) + time
            for pos, slice_ins in enumerate(partition_map[partition]["instances"]):
                if slice_ins == instance:
                    slices_t[pos] = min_time_next_task
        elif type == "exchange":
            slice_0, slice_1 = slices_t[0], slices_t[1]
            slices_t[0], slices_t[1] = slices_t[2], slices_t[3]
            slices_t[2], slices_t[3] = slice_0, slice_1
            
    return max(slices_t)