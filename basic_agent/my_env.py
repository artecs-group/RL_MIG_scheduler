import gymnasium as gym
import ray
import numpy as np
from ray.rllib.algorithms import ppo
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Box
from utils import partition_map, canonical_sort_tasks, print_obs



class SchedEnv(gym.Env):
    def __init__(self, env_config):
        self.N = env_config["N"]
        self.M = env_config["M"]
        self.observation_space = Dict({
            "observations": Dict({
                "partition": Discrete(19), # 19 particiones posibles
                "slices_t": MultiDiscrete([(self.M + 1)] * 7), # Cuanto le queda (de 0 a M) a cada slice
                "ready_tasks": MultiDiscrete([[(self.M + 1)] * 5 + [(self.N + 1)]] * self.N) # El tipo de tarea (tiempo que tarda con cada una de los 5 tamaños e instancia) de las hasta N tareas libres, y la cantidad de tareas que hay de ese tipo (componente 6)
            }),
            "action_mask": Box(0, 1, shape=(1 + 19 + 7 * self.N,)) # Máscara de acciones válidas
        })

        self.action_space = Discrete(1 + 19 + 7 * self.N) # 1 accion esperar, 19 acciones de configuración, y 7*N acciones de asignar tarea


    def _get_action_mask(self, current_partition, slices_t, ready_tasks):
        # Reconfiguraciones válidas
        reconfig_mask = [1] * 19
        # Prohibido reconfigurar a la partición actual
        reconfig_mask[current_partition - 1] = 0
        # Prohibido reconfigurar a particiones en que haya que cambiar slices en uso
        for future_partition in range(1, 20):
            for curr_slice in range(7):
                if partition_map[current_partition]["slices"][curr_slice] != partition_map[future_partition]["slices"][curr_slice] and slices_t[curr_slice] > 0:
                    reconfig_mask[future_partition - 1] = 0

        # Acciones válidas sobre tareas ready
        select_ready_task = [1] * 7 * self.N
        # Prohibido coger una tarea para ejecutar en la GPU más allá de los tipos disponibles
        for i, task in enumerate(ready_tasks):
            n_availables = task[-1]
            # Si en ready hay una no disponible todas las de detrás no están disponibles por el orden canónico
            if n_availables == 0:
                for j in range(i*7, 7 * self.N):
                    select_ready_task[j] = 0
                break
        
        # Prohibido asignar la tarea ready a una instancia con indice superior a la cantidad de instancias de la partición actual
        first_forbidden_instance = len(partition_map[current_partition]["instances"])
        for task in range(self.N):
            for instance in range(first_forbidden_instance, 7):
                select_ready_task[task * 7 + instance] = 0

        first_instance_slice = 0
        # Prohibido asignar la tarea ready a una instancia con índice <= cantidad de instancias de la partición actual
        for instance_index in range(first_forbidden_instance):
            if slices_t[first_instance_slice] > 0:
                for task in range(self.N):
                    select_ready_task[task * 7 + instance_index] = 0
            first_instance_slice += partition_map[current_partition]["instances"][instance_index]

        return [1] + reconfig_mask + select_ready_task
        

    def reset(self, seed = None, options = None):

        num_ready = self.N if np.random.rand() < 0.8 else np.random.randint(1, self.N)
        self.pending_tasks = [sorted(np.random.randint(1, self.M + 1, size=5), reverse=True) for _ in range(num_ready)]
        init_partition = 1 # Por ahora, consideramos que se empieza en la partición 1 (una sola instancia de tamaño 7 siempre)
        init_slice_t = [0] * 7 # Consideramos que todos los slices están libres al principio
        ready_tasks_canonical = canonical_sort_tasks(self.M, self.pending_tasks[:self.N]) # Las siguientes N tareas pendientes, se ordenan canónicamente y colocando como 6ª componente la cantidad de veces que se repite
        # Usamos el 0 para rellenar posiciones vacías en la representación de tareas ready
        init_ready_tasks = ready_tasks_canonical + [[0] * 6] * (self.N - len(ready_tasks_canonical)) # Rellenamos con arrays de 6 ceros hasta N
        




        initial_obs = {
            "observations":{
                "partition": init_partition,
                "slices_t": np.array(init_slice_t),
                "ready_tasks": np.array(init_ready_tasks)
            },
            "action_mask": self._get_action_mask(init_partition, init_slice_t, init_ready_tasks)
        }

        print_obs(initial_obs)

        return initial_obs, {}
    
    def step(self, action):
        pass




env_example = SchedEnv({"N": 5, "M": 3})
#obs = env_example.observation_space.sample()
initial_obs, _ = env_example.reset(seed=0)
#print(initial_obs)