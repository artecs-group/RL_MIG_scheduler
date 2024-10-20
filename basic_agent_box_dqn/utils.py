from collections import Counter
import random
from task_times import generate_tasks

# Mapa de número de partición a sus instancias
partition_map = {
    1: {"slices": ["7"] * 7, "sizes": [7], "instances" : [0,0,0,0,0,0,0]}, 
    2: {"slices": ["4"] * 4 + ["3"] * 3, "sizes": [4, 3], "instances" : [0,0,0,0,1,1,1]}, 
    3: {"slices": ["4"] * 4 + ["2"] * 2 + ["1"], "sizes": [4, 2, 1], "instances" : [0,0,0,0,1,1,2]}, 
    4: {"slices": ["4"] * 4 + ["1"] * 3, "sizes": [4, 1, 1, 1], "instances" : [0,0,0,0,1,2,3]}, 
    5: {"slices": ["3"] * 4 + ["3"] * 3, "sizes": [3, 3], "instances" : [0,0,0,0,1,1,1]}, 
    6: {"slices": ["3"] * 4 + ["2"] * 2 + ["1"], "sizes": [3, 2, 1], "instances" : [0,0,0,0,1,1,2]}, 
    7: {"slices": ["3"] * 4 + ["1"] * 3, "sizes": [3, 1, 1, 1], "instances" : [0,0,0,0,1,2,3]}, 
    8: {"slices": ["2"] * 4 + ["3"] * 3, "sizes": [2, 2, 3], "instances" : [0,0,1,1,2,2,2]}, 
    9: {"slices": ["2"] * 6 + ["1"], "sizes": [2, 2, 2, 1], "instances" : [0,0,1,1,2,2,3]}, 
    10: {"slices": ["2"] * 4 + ["1"] * 3, "sizes": [2, 2, 1, 1, 1], "instances" : [0,0,1,1,2,3,4]}, 
    11: {"slices": ["2"] * 2 + ["1"] * 2 + ["3"] * 3, "sizes": [2, 1, 1, 3], "instances" : [0,0,1,2,3,3,3]}, 
    12: {"slices": ["2"] * 2 + ["1"] * 2 + ["2"] * 2 + ["1"], "sizes": [2, 1, 1, 2, 1], "instances" : [0,0,1,2,3,3,4]}, 
    13: {"slices": ["2"] * 2 + ["1"] * 5, "sizes": [2, 1, 1, 1, 1, 1], "instances" : [0,0,1,2,3,4,5]}, 
    14: {"slices": ["1"] * 2 + ["2"] * 2 + ["3"] * 3, "sizes": [1, 1, 2, 3], "instances" : [0,1,2,2,3,3,3]}, 
    15: {"slices": ["1"] * 2 + ["2"] * 4 + ["1"], "sizes": [1, 1, 2, 2, 1], "instances" : [0,1,2,2,3,3,4]}, 
    16: {"slices": ["1"] * 2 + ["2"] * 2 + ["1"] * 3, "sizes": [1, 1, 2, 1, 1, 1], "instances" : [0,1,2,2,3,4,5]}, 
    17: {"slices": ["1"] * 4 + ["3"] * 3, "sizes": [1, 1, 1, 1, 3], "instances" : [0,1,2,3,4,4,4]}, 
    18: {"slices": ["1"] * 4 + ["2"] * 2 + ["1"], "sizes": [1, 1, 1, 1, 2, 1], "instances" : [0,1,2,3,4,4,5]}, 
    19: {"slices": ["1"] * 7, "sizes": [1, 1, 1, 1, 1, 1, 1], "instances" : [0,1,2,3,4,5,6]}, 
}


# Mapa de tamaño de instancia a posición de array en que se codifica
instance_size_map = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4}


# Pasa a entero desde M-ário como representación
def type_num_task(M, task):
    return sum(time * (M ** i) for i, time in enumerate(task[::-1]))

def canonical_sort_tasks(M, tasks):
    # Pongo el número de tipo de tarea a cada una
    numbered_tasks = [(type_num_task(M, task), task) for task in tasks]
    # Cuento repeticiones de cada tipo
    repeticiones = Counter(map(lambda v: v[0], numbered_tasks))
    # Elimino repetidos (ya he contado cuantos había de cada tipo)
    numbered_tasks = dict(numbered_tasks)
    # Ordeno por tipo de tarea
    canonical_tasks = sorted(numbered_tasks.items(), key=lambda x: x[0])
    # Añado como última componente de cada tipo la cantidad de veces que se repite
    canonical_tasks = [task + [repeticiones[type]] for type, task in canonical_tasks]     
    return canonical_tasks


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
    print("\tReconfiguración:", action_mask[1:20])
    for i, task in enumerate(state["ready_tasks"]):
        print("\tPut task in instance:", action_mask[20 + i * 7: 20 + (i+1) * 7])
    print("-----------")

def _action_to_str(action):
    action = int(action)
    if action == 0:
        return "Wait"
    elif action < 20:
        return f"Reconfigure to {partition_map[action]['sizes']}"
    else:
        task = (action - 20) // 7
        instance = (action - 20) % 7
        return f"Put task {task} in instance {instance}"
    
def time_discretization(ready_tasks, M):
    max_time = max(max(task) for task in ready_tasks)
    time_step = max_time / M
    return [[max(1, round(time / time_step)) for time in task] for task in ready_tasks] 


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
    elif type_tasks == "mix_scaling": 
        # ready_tasks_good = get_ready_tasks(type_tasks="good_scaling", N = N // 2)
        # ready_tasks_bad = get_ready_tasks(type_tasks="bad_scaling", N = N // 2 if N % 2 == 0 else N // 2 + 1)
        scale_percs = [0,0,0,0,1]
        N_good = N // 2
        # n_scale= {ins_size: int(perc*N_good) for ins_size, perc in zip(instance_sizes, scale_percs)}
        # n_scale = _n_scale_padding(n_scale, instance_sizes[::-1], N_good)
        # ready_tasks_good = generate_tasks(instance_sizes=instance_sizes, n_scale=n_scale, device="A100", perc_membound=100, times_range=[90,100])
        ready_tasks_good = [[random.uniform(90, 100)] for _ in range(N_good)]
        for task_times in ready_tasks_good:
            for _ in range(3):
                last_time = task_times[-1]
                task_times.append(last_time * random.uniform(0.97, 1))
            task_times.append(random.uniform(0.1, 5))

        scale_percs = [1,0,0,0,0]
        N_bad = N // 2 if N % 2 == 0 else N // 2 + 1
        n_scale= {ins_size: int(perc*N_bad) for ins_size, perc in zip(instance_sizes, scale_percs)}
        n_scale = _n_scale_padding(n_scale, instance_sizes, N_bad)
        ready_tasks_bad = generate_tasks(instance_sizes=instance_sizes, n_scale=n_scale, device="A100", perc_membound=0, times_range=[90,100])

        ready_tasks = ready_tasks_good + ready_tasks_bad
    return ready_tasks