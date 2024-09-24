import gymnasium as gym
import ray
import numpy as np
from ray.rllib.algorithms import ppo
from gymnasium.spaces import Dict, Discrete, MultiDiscrete
from collections import Counter


class SchedEnv(gym.Env):
    def __init__(self, env_config):
        self.N = env_config["N"]
        self.M = env_config["M"]
        self.observation_space = Dict({
            "partition": Discrete(19), # 19 particiones posibles
            "slices_t": MultiDiscrete([(self.M + 1)] * 7), # Cuanto le queda (de 0 a M) a cada slice
            "ready_tasks": MultiDiscrete([[(self.M + 1)] * 5 + [(self.N + 1)]] * self.N) # El tipo de tarea (tiempo que tarda con cada una de los 5 tamaños e instancia) de las hasta N tareas libres, y la cantidad de tareas que hay de ese tipo (componente 6)
        })

        self.action_space = Discrete(1 + 19 + 7 * self.N) # 1 accion esperar, 19 acciones de configuración, y 7*N acciones de asignar tarea

    # Pasa a entero desde M-ário como representación
    def _type_num_task(self, task):
        return sum(time * (self.M ** i) for i, time in enumerate(task[::-1]))
    
    def _canonical_sort_tasks(self, tasks):
        # Pongo el número de tipo de tarea a cada una
        numbered_tasks = [(self._type_num_task(task), task) for task in tasks]
        # Cuento repeticiones de cada tipo
        repeticiones = Counter(map(lambda v: v[0], numbered_tasks))
        # Elimino repetidos (ya he contado cuantos había de cada tipo)
        numbered_tasks = dict(numbered_tasks)
        # Ordeno por tipo de tarea
        canonical_tasks = sorted(numbered_tasks.items(), key=lambda x: x[0])
        # Añado como última componente de cada tipo la cantidad de veces que se repite
        canonical_tasks = [task + [repeticiones[type]] for type, task in canonical_tasks]     
        return canonical_tasks

        
    def reset(self, seed = None, options = None):

        num_ready = self.N if np.random.rand() < 0.8 else np.random.randint(1, self.N)
        self.pending_tasks = [sorted(np.random.randint(1, self.M + 1, size=5), reverse=True) for _ in range(num_ready)]
        init_partition = np.random.randint(1, 20) # Particion inicial aleatoria
        init_slice_t = [0] * 7 # Consideramos que todos los slices están libres al principio
        ready_tasks_canonical = self._canonical_sort_tasks(self.pending_tasks[:self.N]) # Las siguientes N tareas pendientes, se ordenan canónicamente y colocando como 6ª componente la cantidad de veces que se repite
        # Usamos el 0 para rellenar posiciones vacías en la representación de tareas ready
        init_ready_tasks = [ready_tasks_canonical + [[0] * 6] * (self.N - len(ready_tasks_canonical))] # Rellenamos con arrays de 6 ceros hasta N
        initial_obs = {
            "partition": init_partition,
            "slices_t": np.array(init_slice_t),
            "ready_tasks": np.array(init_ready_tasks)
        }

        return initial_obs, {}




env_example = SchedEnv({"N": 5, "M": 3})
#obs = env_example.observation_space.sample()
initial_obs, _ = env_example.reset(seed=0)
print(initial_obs)