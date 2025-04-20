import random
import numpy as np
import pandas as pd

def read_task_rodinia():
    df_1g = pd.read_csv("rodinia_times/1g_A100.csv")
    col_times = [col for col in df_1g.columns if col != 'prueba']
    df_1g["mean"] = df_1g[col_times].mean(axis=1)
    df_1g = df_1g[["prueba", "mean"]]

    df_2g = pd.read_csv("rodinia_times/2g_A100.csv")
    df_2g["mean"] = df_2g[col_times].mean(axis=1)
    df_2g = df_2g[["prueba", "mean"]]

    df_3g = pd.read_csv("rodinia_times/3g_A100.csv")
    df_3g["mean"] = df_3g[col_times].mean(axis=1)
    df_3g = df_3g[["prueba", "mean"]]

    df_4g = pd.read_csv("rodinia_times/4g_A100.csv")
    df_4g["mean"] = df_4g[col_times].mean(axis=1)
    df_4g = df_4g[["prueba", "mean"]]

    df_7g = pd.read_csv("rodinia_times/7g_A100.csv")
    df_7g["mean"] = df_7g[col_times].mean(axis=1)
    df_7g = df_7g[["prueba", "mean"]]

    tiempos_prueba = lambda prueba_name: [list(df["mean"][df["prueba"] == prueba_name])[0] for df in [df_1g, df_2g, df_3g, df_4g, df_7g]]
    times = [[(index, slice, time) for slice, time in zip([1,2,3,4,7], tiempos_prueba(prueba))] for index, prueba in enumerate(df_1g["prueba"])]
    return times, list(df_1g["prueba"])


def get_input_config():
    repetitions = int(input("Repetitions: "))
    available_devices = ["A30", "A100"]
    while True:
        device = input("Goal device (A30 or A100): ")
        if device in available_devices:
            break

    instance_sizes = [1,2,4] if device == "A30" else [1,2,3,4,7] if device == "A100" else None
    n_scale = {}
    for size in instance_sizes:
        n_scale[size] = int(input(f"Number scale until {size} slices: "))
    perc_membound = float(input("Percentage memory bound (0-100%): "))
    return repetitions, instance_sizes, n_scale, perc_membound, device

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
                    next_time = (size - 1 + np.clip(np.random.normal(-0.3, 0.3), -0.5, 0)) / size * last_time
                    if random.random() <= 0.025:
                        super_linear_grow = False
                else:
                    next_time = (size - 1 + np.clip(np.random.normal(0.05, 0.05), 0, 0.1)) / size * last_time

                times_instance_scale_size[i].append((size, next_time))
        times += times_instance_scale_size
    # Remove times of instance sizes not valid
    times = [[time for slices, time in task_times if slices in instance_sizes] for task_times in times]
    # # Aux for cheking agent selection
    # for i in range(len(times)):
    #     times[i][1] = times[i][0]


    return times