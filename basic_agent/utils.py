from collections import Counter

# Mapa de número de partición a sus instancias
partition_map = {
    1: {"slices": ["7"] * 7, "instances": [7]}, 
    2: {"slices": ["4"] * 4 + ["3"] * 3, "instances": [4, 3]}, 
    3: {"slices": ["4"] * 4 + ["2"] * 2 + ["1"], "instances": [4, 2, 1]}, 
    4: {"slices": ["4"] * 4 + ["1"] * 3, "instances": [4, 1, 1, 1]}, 
    5: {"slices": ["3"] * 4 + ["3"] * 3, "instances": [3, 3]}, 
    6: {"slices": ["3"] * 4 + ["2"] * 2 + ["1"], "instances": [3, 2, 1]}, 
    7: {"slices": ["3"] * 4 + ["1"] * 3, "instances": [3, 1, 1, 1]}, 
    8: {"slices": ["2"] * 4 + ["3"] * 3, "instances": [2, 2, 3]}, 
    9: {"slices": ["2"] * 6 + ["1"], "instances": [2, 2, 2, 1]}, 
    10: {"slices": ["2"] * 4 + ["1"] * 3, "instances": [2, 2, 1, 1, 1]}, 
    11: {"slices": ["2"] * 2 + ["1"] * 2 + ["3"] * 3, "instances": [2, 1, 1, 3]}, 
    12: {"slices": ["2"] * 2 + ["1"] * 2 + ["2"] * 2 + ["1"], "instances": [2, 1, 1, 2, 1]}, 
    13: {"slices": ["2"] * 2 + ["1"] * 5, "instances": [2, 1, 1, 1, 1, 1]}, 
    14: {"slices": ["1"] * 2 + ["2"] * 2 + ["3"] * 3, "instances": [1, 1, 2, 3]}, 
    15: {"slices": ["1"] * 2 + ["2"] * 4 + ["1"], "instances": [1, 1, 2, 2, 1]}, 
    16: {"slices": ["1"] * 2 + ["2"] * 2 + ["1"] * 3, "instances": [1, 1, 2, 1, 1, 1]}, 
    17: {"slices": ["1"] * 4 + ["3"] * 3, "instances": [1, 1, 1, 1, 3]}, 
    18: {"slices": ["1"] * 4 + ["2"] * 2 + ["1"], "instances": [1, 1, 1, 1, 2, 1]}, 
    19: {"slices": ["1"] * 7, "instances": [1, 1, 1, 1, 1, 1, 1]}, 
}


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


def print_obs(obs):
    state = obs["observations"]
    action_mask = obs["action_mask"]
    print("-----------")
    print("State:")
    print("\tPartition:", partition_map[state["partition"]]["instances"])
    print("\tSlices:", state["slices_t"])
    for task in state["ready_tasks"]:
        if task[-1] != 0:
            print("\tTask type:", task[:5], "number", task[5])
    print("Action mask:")
    print("\tEsperar:", action_mask[0])
    print("\tReconfiguración:", action_mask[1:20])
    for i, task in enumerate(state["ready_tasks"]):
        print("\tTipo tarea:", action_mask[20 + i * 7: 20 + (i+1) * 7])
    print("-----------")