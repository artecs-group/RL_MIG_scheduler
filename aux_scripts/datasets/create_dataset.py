import random
import numpy as np
import pickle

def generate_tasks(instance_sizes, n_scale, device, perc_membound = 0, times_range=[1,100]):
    n_slices = instance_sizes[-1]
    times = []
    for scale_size in instance_sizes:
        n_instance_scale_size = n_scale[scale_size]
        times_instance_scale_size =  [[(1, random.uniform(times_range[0], times_range[1]))] for _ in range(n_instance_scale_size)]
        n_mem_bound = n_instance_scale_size * perc_membound // 100
        for i in range(n_instance_scale_size):
            super_linear_grow = i < n_mem_bound
            for size in range(2, n_slices+1):
                _, last_time = times_instance_scale_size[i][-1]
                # Escala mal
                if size > scale_size or super_linear_grow and device == "A100" and size == 4:
                    next_time = (size - 1 + np.clip(np.random.normal(0.75, 0.25), 0.5, 1)) / size * last_time
                # Si sigue siendo memory bound y hemos escalado en memoria (de 3 a 4 slices no se escala en A100)
                elif super_linear_grow and size != 4:
                    next_time = (size - 1 + np.clip(np.random.normal(-0.25, 0.25), -0.3, 0)) / size * last_time
                    if random.random() <= 0.1:
                        super_linear_grow = False
                else:
                    next_time = (size - 1 + np.clip(np.random.normal(0.05, 0.05), 0, 0.1)) / size * last_time

                times_instance_scale_size[i].append((size, next_time))
        times += times_instance_scale_size
    # Remove times of instance sizes not valid
    times = [[(index, slices, time) for slices, time in task_times if slices in instance_sizes] for index, task_times in enumerate(times)]
    #pprint(times)
    return times, instance_sizes


def create_dataset(workload, num_datasets):
    instance_sizes = [1,2,3,4,7]
    device = "A100"
    dataset = []
    for num_dataset in range(num_datasets):
        if workload == "good_scaling":
            n_scale = {1: 0, 2: 0, 3: 0, 4: 50, 7: 50}
            perc_membound = 75
            times_range = [90, 100]
        elif workload == "bad_scaling":
            n_scale = {1: 50, 2: 50, 3: 0, 4: 0, 7: 0}
            perc_membound = 25
            times_range = [90, 100]
        elif workload == "mix_scaling_uniform":
            n_scale = {1: 20, 2: 20, 3: 20, 4: 20, 7: 20}
            perc_membound = 50
            times_range = [90, 100]
        elif workload == "mix_scaling_extreme":
            n_scale = {1: 45, 2: 5, 3: 0, 4: 5, 7: 45}
            perc_membound = 50
            times_range = [90, 100]
        elif workload == "wide_times":
            n_scale = {1: 20, 2: 20, 3: 20, 4: 20, 7: 20}
            perc_membound = 50
            times_range = [1, 100]
        else:
            return      
        
        times, _ = generate_tasks(instance_sizes, n_scale, device, perc_membound, times_range)
        dataset.append(times)
        print(num_dataset)
    with open(f"dataset_{workload}.pkl", "wb") as f:
        pickle.dump(dataset, f)



if __name__ == "__main__":
    workload = input("Dataset type: ")
    num_datasets =int(input("Num datasets: "))
    create_dataset(workload, num_datasets)
