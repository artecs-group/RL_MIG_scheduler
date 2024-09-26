from collections import Counter
from matplotlib import patches
import matplotlib.pyplot as plt

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


def graphic_obs(M, obs, num_task_slices):

    colors = plt.cm.tab20.colors
    figsize = (30, 5)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.set_size_inches(12, 5)
    state = obs["observations"]
    action_mask = obs["action_mask"]
    slice_i = 0
    for instance_size in partition_map[state["partition"]]["sizes"]:
        # Representa los 7 valores de state["slices_t"] en una gráfica de barras
        rect = patches.Rectangle((slice_i, 0), instance_size, state["slices_t"][slice_i], alpha = 0.55,\
                                        linewidth = 1, facecolor = colors[num_task_slices[slice_i] % len(colors)], edgecolor = 'black')
        ax1.add_patch(rect)
        slice_i += instance_size
    ax1.set_ylim([0, M])
    ax1.set_xlim([0, 7])
    ax1.set_xlabel("Slices")
    ax1.set_ylabel("Time")
    ax1.set_yticks(range(0, M+1))
    ax1.set_title(f"GPU Partition {partition_map[state['partition']]['sizes']}")

    slices_l= [1,2,3,4,7]
    # Create a bar chart for each task in state["ready_tasks"]
    for i, task in enumerate(state["ready_tasks"]):
        # Calculate the x positions for the bars
        x = [j + i * 5 for j in range(5)]
        # Create the bar chart
        ax2.bar(x, task[:5], color=colors[i % len(colors)], alpha=0.55)
        if task[-1] == 0:
            continue
        for j, slice_pos in enumerate(x):
            ax2.text(slice_pos, -M/100, slices_l[j], ha='center', va='top')
        ax2.text(2 + i * 5, -M/20, str(f"{task[-1]} {'task'if task[-1] == 1 else 'tasks'}"), ha='center', va='top')
        
    # Set y labels for the bar chart
    ax2.set_ylabel("Time")
    ax1.set_ylim([0, M])
    # Set the title for the bar chart
    ax2.set_title("Ready tasks")
    # Remove the ticks on the x-axis of ax2
    ax2.set_xticks([])
    ax2.set_yticks(range(0, M+1))

    # Add text below the figures
    wait, reconfigure, n_task = action_mask[0], action_mask[1:20], action_mask[20:]
    plt.subplots_adjust(left=0.1, right=0.9)

    # Remove axes and labels from ax3
    ax3.axis('off')

    ax3.set_title("Posible actions")
    # Add text to ax3
    ax3.text(0.5, 0.95, "Wait" if wait else "", ha='center', va='center', fontsize=12)

    ax3.text(0, 0.9, "Reconfigs:", ha='left', va='center', fontsize=12)
    height = 0.85
    for part, reconfig in enumerate(reconfigure):
        if reconfig == 1:
            ax3.text(0.5, height, str(partition_map[part+1]["sizes"]), ha='center', va='center', fontsize=12)
            height -= 0.05

    # # Draw rectangles on ax3
    # rect1 = patches.Rectangle((0.1, 0.1), 0.3, 0.3, linewidth=1, edgecolor='black', facecolor='red')
    # rect2 = patches.Rectangle((0.6, 0.1), 0.3, 0.3, linewidth=1, edgecolor='black', facecolor='blue')
    # ax3.add_patch(rect1)
    # ax3.add_patch(rect2)


    plt.show()
    
